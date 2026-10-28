-- Find the people a misspelt name was probably meant to be.
--
-- A researcher working from a handwritten register types what they read, and what they read
-- is often a letter or two off. Today that returns "no matching records", which is both
-- wrong and unhelpful: the person is in the list, spelled slightly differently, and the
-- researcher has no way to know whether they mistyped or the record simply is not there.
--
-- Matching is by edit distance over row_data_normalized.names -- the token index the
-- backfill already builds, holding every recorded form of a person's name: canonical,
-- alternate, misspelt, Indigenous, and the spellings of each. So a near-miss against any
-- recorded form is found, not just against the display name.
--
-- Distance is per query token: each word the user typed is scored against its closest token
-- on the row, and the row's distance is the sum. MAX(best) <= p_max_distance keeps one
-- badly wrong word from being absorbed by other good ones -- "Albert Xylophone" is not a
-- near miss for "Albert Penance" just because "Albert" matched.
--
-- The caller decides what to do with the distance: 1 is a typo, and worth answering
-- directly with the correction stated; more than that is a question worth asking.

CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

CREATE OR REPLACE FUNCTION assistant_api.suggest_name_matches(
    p_principal_id text,
    p_requested_file_ids integer[],
    p_can_use_private boolean,
    p_query text,
    p_max_distance integer DEFAULT 2,
    p_limit integer DEFAULT 10
) RETURNS TABLE(
    file_id integer,
    source_row_id bigint,
    display_name text,
    distance integer
)
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path TO 'assistant_api', 'assistant', 'public'
AS $function$
    WITH query_tokens AS (
        SELECT DISTINCT lower(btrim(token)) AS token
        FROM regexp_split_to_table(COALESCE(p_query, ''), '[[:space:]]+') AS token
        WHERE btrim(token) <> ''
    ),
    candidate AS (
        SELECT r.file_id, r.source_row_id, r.row_data_normalized,
               -- The name as recorded, not the casefolded index form: this is read back to
               -- a person, and "albert penance" is not how the register writes it.
               COALESCE(
                   NULLIF(btrim(r.row_data_normalized #>> '{canonical,display_name}'), ''),
                   r.canonical_name
               ) AS display_name
        FROM assistant_api.v_current_records r
        WHERE r.file_id = ANY (
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        )
    ),
    scored AS (
        SELECT
            candidate.file_id,
            candidate.source_row_id,
            candidate.display_name,
            query_tokens.token,
            MIN(
                levenshtein_less_equal(
                    query_tokens.token,
                    lower(recorded.token),
                    GREATEST(COALESCE(p_max_distance, 2), 1)
                )
            ) AS best
        FROM candidate
        CROSS JOIN query_tokens
        CROSS JOIN LATERAL jsonb_array_elements_text(
            CASE
                WHEN jsonb_typeof(candidate.row_data_normalized->'names') = 'array'
                THEN candidate.row_data_normalized->'names'
                ELSE '[]'::jsonb
            END
        ) AS recorded(token)
        GROUP BY 1, 2, 3, 4
    )
    SELECT
        scored.file_id,
        scored.source_row_id,
        scored.display_name,
        SUM(scored.best)::INTEGER AS distance
    FROM scored
    GROUP BY 1, 2, 3
    HAVING MAX(scored.best) <= GREATEST(COALESCE(p_max_distance, 2), 1)
       AND COUNT(*) = (SELECT COUNT(*) FROM query_tokens)
    ORDER BY SUM(scored.best), lower(scored.display_name), scored.source_row_id
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 10), 25), 1);
$function$;

GRANT EXECUTE ON FUNCTION assistant_api.suggest_name_matches(
    text, integer[], boolean, text, integer, integer
) TO assistant_runtime, assistant_gateway_owner;
