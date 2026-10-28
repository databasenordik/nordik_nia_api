-- Register every currently public column through the semantic catalog.  The
-- assistant still reads normalized rows through assistant_api; this does not
-- grant the runtime direct table access.
INSERT INTO assistant.field_registry (
    file_id, semantic_field, human_label, aliases, canonical_json_path, raw_json_keys,
    semantic_type, allowed_operators, searchable, aggregatable, sortable, quoteable,
    evidence_allowed, raw_fetch_allowed
)
SELECT * FROM (VALUES
    (49, 'student_name', 'Student name', ARRAY['name','student','full name'], 'canonical.display_name', ARRAY['STUDENT NAME','Student Name'], 'text', ARRAY['EQUALS','STARTS_WITH','CONTAINS','FULL_TEXT_SEARCH','FUZZY_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, TRUE, TRUE, FALSE),
    (49, 'first_name', 'First name', ARRAY['first names','given name'], 'canonical.first_name', ARRAY['First Names'], 'text', ARRAY['EQUALS','STARTS_WITH','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
    (49, 'middle_names', 'Middle names', ARRAY['middle name'], 'canonical.middle_names', ARRAY['Middle Names'], 'text', ARRAY['EQUALS','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
    (49, 'last_name', 'Last name', ARRAY['last names','surname','family name'], 'canonical.last_name', ARRAY['Last Names'], 'text', ARRAY['EQUALS','STARTS_WITH','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
    (49, 'indigenous_name', 'Indigenous name / Spirit name', ARRAY['indian name','spirit name','indigenous name'], 'canonical.indigenous_name', ARRAY['Indigenous Name/Spirit Name'], 'text', ARRAY['EQUALS','CONTAINS','IS_KNOWN','IS_UNKNOWN','FULL_TEXT_SEARCH'], TRUE, FALSE, TRUE, TRUE, TRUE, FALSE),
    (49, 'community', 'First Nation / Community', ARRAY['reserve','first nation','community'], 'canonical.community', ARRAY['First Nation/Community','Community'], 'entity', ARRAY['EQUALS','IN','CONTAINS','FUZZY_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (49, 'admitted_date', 'Admitted date', ARRAY['admitted','admission date','date admitted'], 'canonical.dates.admitted', ARRAY['Admitted','Date of Admission'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (49, 'discharged_date', 'Discharged date', ARRAY['discharged','discharge date'], 'canonical.dates.discharged', ARRAY['Discharged'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (49, 'birth_date', 'Date of birth', ARRAY['birth date','born','dob'], 'canonical.dates.birth', ARRAY['Date of Birth'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (49, 'deceased_status', 'Deceased status', ARRAY['deceased','died','death status'], 'canonical.deceased_status', ARRAY['Deceased?','Deceased'], 'boolean', ARRAY['IS_TRUE','IS_FALSE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (49, 'age', 'Age', ARRAY['student age'], 'fields.Age', ARRAY['Age'], 'number', ARRAY['EQUALS','GREATER_THAN','LESS_THAN','NUMBER_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (49, 'student_number', 'Student number', ARRAY['student id','record number'], 'canonical.student_number', ARRAY['Student Number'], 'text', ARRAY['EQUALS','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
    (49, 'parents_names', 'Parents names', ARRAY['parents','parent names','family'], 'canonical.parents_names', ARRAY['Parents Names'], 'text', ARRAY['EQUALS','CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (49, 'siblings', 'Siblings', ARRAY['brothers','sisters'], 'fields.Siblings', ARRAY['Siblings'], 'text', ARRAY['EQUALS','CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (49, 'notes', 'Notes', ARRAY['note','remarks'], 'chat.narrative_bundle.notes', ARRAY['Notes'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','GET_QUOTE','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (49, 'death_details', 'Death details', ARRAY['death information','death notes'], 'fields.Death details', ARRAY['Death details'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','GET_QUOTE','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (49, 'additional_information', 'Additional information', ARRAY['additional info','other information'], 'fields.Additional Information', ARRAY['Additional Information'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','GET_QUOTE','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (49, 'mapping_location', 'Mapping location', ARRAY['map location','location'], 'canonical.mapping_location', ARRAY['Mapping Location'], 'entity', ARRAY['EQUALS','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (49, 'latitude', 'Latitude', ARRAY['lat'], 'canonical.lat', ARRAY['Lat'], 'number', ARRAY['EQUALS','GREATER_THAN','LESS_THAN','NUMBER_RANGE','IS_KNOWN','IS_UNKNOWN'], FALSE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (49, 'longitude', 'Longitude', ARRAY['lng','lon'], 'canonical.lng', ARRAY['Lng'], 'number', ARRAY['EQUALS','GREATER_THAN','LESS_THAN','NUMBER_RANGE','IS_KNOWN','IS_UNKNOWN'], FALSE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (49, 'photos', 'Photos', ARRAY['photo','images'], 'fields.Photos', ARRAY['Photos'], 'text', ARRAY['CONTAINS','IS_KNOWN','IS_UNKNOWN'], FALSE, FALSE, FALSE, FALSE, TRUE, FALSE),
    (49, 'documents', 'Documents', ARRAY['document','files'], 'fields.Documents', ARRAY['Documents'], 'text', ARRAY['CONTAINS','IS_KNOWN','IS_UNKNOWN'], FALSE, FALSE, FALSE, FALSE, TRUE, FALSE),

    (91, 'student_name', 'Student name', ARRAY['name','child name','full name'], 'canonical.display_name', ARRAY['STUDENT NAME'], 'text', ARRAY['EQUALS','STARTS_WITH','CONTAINS','FULL_TEXT_SEARCH','FUZZY_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, TRUE, TRUE, FALSE),
    (91, 'cause_of_death', 'Cause of death', ARRAY['cause','causes of death','reason of death','reasons of death'], 'canonical.cause_of_death', ARRAY['CAUSE OF DEATH'], 'entity', ARRAY['EQUALS','IN','CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, TRUE, TRUE, FALSE),
    (91, 'death_factor', 'Death factor', ARRAY['factor'], 'fields.DEATH FACTOR', ARRAY['DEATH FACTOR'], 'entity', ARRAY['EQUALS','IN','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, TRUE, TRUE, FALSE),
    (91, 'birth_date', 'Date of birth', ARRAY['birth date','born','dob'], 'canonical.dates.birth', ARRAY['DATE OF BIRTH'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'death_date', 'Date of death', ARRAY['death date','died','dod'], 'canonical.dates.death', ARRAY['DATE OF DEATH'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'burial_date', 'Date of burial', ARRAY['burial date','buried'], 'fields.DATE OF BURIAL', ARRAY['DATE OF BURIAL'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'first_admitted_date', 'Date first admitted', ARRAY['first admitted','admission date','date admitted'], 'fields.DATE FIRST ADMITTED', ARRAY['DATE FIRST ADMITTED'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'discharged_date', 'Date of discharge', ARRAY['discharge date','discharged'], 'fields.DATE OF DISCHARGE', ARRAY['DATE OF DISCHARGE'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'death_registration_date', 'Date of death registration', ARRAY['registration date','death registered'], E'fields.DATE OF DEATH \nREGISTRATION', ARRAY[E'DATE OF DEATH \nREGISTRATION'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'age_at_death', 'Age at death', ARRAY['age','death age'], 'fields.AGE AT DEATH', ARRAY['AGE AT DEATH'], 'number', ARRAY['EQUALS','GREATER_THAN','LESS_THAN','NUMBER_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'gender', 'Gender', ARRAY['sex'], 'fields.GENDER', ARRAY['GENDER'], 'entity', ARRAY['EQUALS','IN','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'nation', 'Nation', ARRAY['first nation','nationality'], 'fields.NATION', ARRAY['NATION'], 'entity', ARRAY['EQUALS','IN','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'community', 'Community / Reserve', ARRAY['reserve','community','first nation community'], 'fields.COMMUNITY/RESERVE', ARRAY['COMMUNITY/RESERVE'], 'entity', ARRAY['EQUALS','IN','CONTAINS','FUZZY_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'school', 'School', ARRAY['school name'], 'canonical.school', ARRAY['SCHOOL'], 'entity', ARRAY['EQUALS','IN','CONTAINS','FUZZY_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'place_of_death', 'Place of death', ARRAY['death place','died at'], 'fields.PLACE OF DEATH', ARRAY['PLACE OF DEATH'], 'entity', ARRAY['EQUALS','IN','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'location_of_death', 'Location of death', ARRAY['death location'], 'fields.LOCATION OF DEATH', ARRAY['LOCATION OF DEATH'], 'entity', ARRAY['EQUALS','IN','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'place_of_burial', 'Place of burial', ARRAY['burial place','buried at','cemetery'], 'fields.PLACE OF BURIAL', ARRAY['PLACE OF BURIAL'], 'entity', ARRAY['EQUALS','IN','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'student_number', 'Student number', ARRAY['student id','record number'], 'canonical.student_number', ARRAY['STUDENT NUMBER'], 'text', ARRAY['EQUALS','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
    (91, 'indigenous_name', 'Indigenous name', ARRAY['indian name','spirit name'], 'fields.INDIAN NAME', ARRAY['INDIAN NAME'], 'text', ARRAY['EQUALS','CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, TRUE, TRUE, FALSE),
    (91, 'parents_names', 'Parents', ARRAY['parents names','family'], 'canonical.parents_names', ARRAY['PARENTS'], 'text', ARRAY['EQUALS','CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (91, 'ancestry_information', 'Ancestry information', ARRAY['ancestry','family history'], 'fields.ANCESTRY INFORMATION', ARRAY['ANCESTRY INFORMATION'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','GET_QUOTE','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (91, 'admission_method', 'Admission method', ARRAY['how admitted'], 'fields.ADMISSION METHOD', ARRAY['ADMISSION METHOD'], 'entity', ARRAY['EQUALS','IN','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'reason_for_discharge', 'Reason for discharge', ARRAY['discharge reason'], 'fields.REASON FOR DISCHARGE', ARRAY['REASON FOR DISCHARGE'], 'entity', ARRAY['EQUALS','IN','CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, TRUE, TRUE, FALSE),
    (91, 'documentation', 'Documentation', ARRAY['documents','evidence'], 'fields.DOCUMENTATION', ARRAY['DOCUMENTATION'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','GET_QUOTE','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (91, 'investigative_notes', 'Investigative notes', ARRAY['investigation notes'], 'fields.INVESTIGATIVE NOTES', ARRAY['INVESTIGATIVE NOTES'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','GET_QUOTE','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (91, 'specific_file_notes', 'Specific file notes / References', ARRAY['file notes','references','specific references'], E'fields.SPECIFIC FILE NOTES/REFERENCES\n', ARRAY[E'SPECIFIC FILE NOTES/REFERENCES\n'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','GET_QUOTE','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (91, 'information_shared_with', 'Information shared with', ARRAY['shared with'], 'fields.INFORMATION SHARED WITH ?', ARRAY['INFORMATION SHARED WITH ?'], 'text', ARRAY['EQUALS','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
    (91, 'death_registration_number', 'Death registration number', ARRAY['registration number'], E'fields.DEATH REGISTRATION \nNUMBER', ARRAY[E'DEATH REGISTRATION \nNUMBER'], 'text', ARRAY['EQUALS','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
    (91, 'other_schools', 'Other schools / Institutions attended', ARRAY['other institutions','other schools'], 'fields.OTHER SCHOOLS/INSTITUTIONS ATTENDED', ARRAY['OTHER SCHOOLS/INSTITUTIONS ATTENDED'], 'text', ARRAY['EQUALS','CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, TRUE, TRUE, FALSE),
    (91, 'other_links', 'Other links', ARRAY['links','urls'], 'fields.Other Links', ARRAY['Other Links'], 'text', ARRAY['CONTAINS','IS_KNOWN','IS_UNKNOWN'], FALSE, FALSE, FALSE, FALSE, TRUE, FALSE),
    (91, 'other_information', 'Other information', ARRAY['other','additional information'], E'fields.Other \n', ARRAY[E'Other \n'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','GET_QUOTE','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (91, 'census_documents_used', 'Census documents used', ARRAY['census used'], 'fields.INDICATED IF CENSUS DOCUMENTS USED', ARRAY['INDICATED IF CENSUS DOCUMENTS USED'], 'boolean', ARRAY['IS_TRUE','IS_FALSE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'census_year', 'Census year used or required', ARRAY['census year'], 'fields.INDICATE CENSUS YEAR USED OR REQUIRED', ARRAY['INDICATE CENSUS YEAR USED OR REQUIRED'], 'number', ARRAY['EQUALS','GREATER_THAN','LESS_THAN','NUMBER_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (91, 'nia_comments', 'Nia comments', ARRAY['nia comment','comments'], 'fields.NIA Comments', ARRAY['NIA Comments'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','GET_QUOTE','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE)
) AS v(
    file_id, semantic_field, human_label, aliases, canonical_json_path, raw_json_keys,
    semantic_type, allowed_operators, searchable, aggregatable, sortable, quoteable,
    evidence_allowed, raw_fetch_allowed
)
ON CONFLICT (file_id, semantic_field) DO UPDATE SET
    human_label = EXCLUDED.human_label,
    aliases = EXCLUDED.aliases,
    canonical_json_path = EXCLUDED.canonical_json_path,
    raw_json_keys = EXCLUDED.raw_json_keys,
    semantic_type = EXCLUDED.semantic_type,
    allowed_operators = EXCLUDED.allowed_operators,
    searchable = EXCLUDED.searchable,
    aggregatable = EXCLUDED.aggregatable,
    sortable = EXCLUDED.sortable,
    quoteable = EXCLUDED.quoteable,
    evidence_allowed = EXCLUDED.evidence_allowed,
    raw_fetch_allowed = EXCLUDED.raw_fetch_allowed;

-- Complete the typed predicate surface advertised by the registry.
CREATE OR REPLACE FUNCTION assistant_api._apply_op(
    p_value TEXT,
    p_operator TEXT,
    p_expected TEXT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    value_year INTEGER;
    expected_year INTEGER;
    years INTEGER[];
    value_number NUMERIC;
    expected_number NUMERIC;
    range_parts TEXT[];
    expected_json JSONB;
BEGIN
    IF p_operator = 'IS_UNKNOWN' THEN
        RETURN p_value IS NULL OR btrim(p_value) = '';
    END IF;
    IF p_operator = 'IS_KNOWN' THEN
        RETURN p_value IS NOT NULL AND btrim(p_value) <> '';
    END IF;
    IF p_operator = 'IS_TRUE' THEN
        RETURN lower(coalesce(p_value, '')) IN ('true','t','1','yes','y','deceased');
    END IF;
    IF p_operator = 'IS_FALSE' THEN
        RETURN lower(coalesce(p_value, '')) IN ('false','f','0','no','n');
    END IF;
    IF p_value IS NULL THEN
        RETURN FALSE;
    END IF;
    IF p_operator = 'EQUALS' THEN
        RETURN lower(btrim(p_value)) = lower(btrim(coalesce(p_expected, '')));
    END IF;
    IF p_operator = 'STARTS_WITH' THEN
        RETURN starts_with(lower(p_value), lower(coalesce(p_expected, '')));
    END IF;
    IF p_operator = 'CONTAINS' THEN
        RETURN position(lower(coalesce(p_expected, '')) IN lower(p_value)) > 0;
    END IF;
    IF p_operator = 'IN' THEN
        BEGIN
            expected_json := p_expected::jsonb;
            RETURN EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(expected_json) item
                WHERE lower(btrim(item)) = lower(btrim(p_value))
            );
        EXCEPTION WHEN OTHERS THEN
            RETURN FALSE;
        END;
    END IF;
    IF p_operator IN ('YEAR_EQUALS','BEFORE','AFTER') THEN
        value_year := substring(p_value from '(1[6-9][0-9]{2}|20[0-9]{2})')::INTEGER;
        expected_year := substring(coalesce(p_expected, '') from '(1[6-9][0-9]{2}|20[0-9]{2})')::INTEGER;
        IF value_year IS NULL OR expected_year IS NULL THEN RETURN FALSE; END IF;
        IF p_operator = 'YEAR_EQUALS' THEN RETURN value_year = expected_year; END IF;
        IF p_operator = 'BEFORE' THEN RETURN value_year < expected_year; END IF;
        RETURN value_year > expected_year;
    END IF;
    IF p_operator = 'DATE_RANGE' THEN
        SELECT array_agg(match[1]::INTEGER)
        INTO years
        FROM regexp_matches(coalesce(p_expected, ''), '(1[6-9][0-9]{2}|20[0-9]{2})', 'g') match;
        value_year := substring(p_value from '(1[6-9][0-9]{2}|20[0-9]{2})')::INTEGER;
        RETURN value_year IS NOT NULL AND cardinality(years) >= 2
            AND value_year BETWEEN years[1] AND years[2];
    END IF;
    IF p_operator IN ('GREATER_THAN','LESS_THAN','NUMBER_RANGE') THEN
        BEGIN
            value_number := NULLIF(regexp_replace(p_value, '[^0-9.-]', '', 'g'), '')::NUMERIC;
            IF p_operator = 'NUMBER_RANGE' THEN
                range_parts := regexp_split_to_array(
                    regexp_replace(coalesce(p_expected, ''), '[^0-9.,-]', '', 'g'), ','
                );
                RETURN cardinality(range_parts) >= 2
                    AND value_number BETWEEN range_parts[1]::NUMERIC AND range_parts[2]::NUMERIC;
            END IF;
            expected_number := NULLIF(regexp_replace(coalesce(p_expected, ''), '[^0-9.-]', '', 'g'), '')::NUMERIC;
            IF p_operator = 'GREATER_THAN' THEN RETURN value_number > expected_number; END IF;
            RETURN value_number < expected_number;
        EXCEPTION WHEN OTHERS THEN
            RETURN FALSE;
        END;
    END IF;
    RETURN FALSE;
END;
$$;

ALTER FUNCTION assistant_api._apply_op(TEXT, TEXT, TEXT) OWNER TO assistant_gateway_owner;
REVOKE ALL ON FUNCTION assistant_api._apply_op(TEXT, TEXT, TEXT) FROM PUBLIC;
