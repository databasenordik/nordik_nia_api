-- Tell the planner what name_search is for, in the catalog rather than the prompt.
--
-- 053 registered the field and it works, but the planner kept choosing student_name for
-- person lookups and "Who is Wm. Fisher?" still found nothing. That is a labelling
-- problem, not a planning bug: both fields said "name", student_name said it first and
-- more plainly, and nothing in the catalog said one of them matches recorded variants
-- while the other matches only the display cell.
--
-- The prompt carries no dataset vocabulary by design -- rules key on catalog properties --
-- so the fix belongs here, where a label and its aliases are data. student_name keeps its
-- own label; it is still the right field for showing a person's name and for an exact
-- match on the display cell.

UPDATE assistant.field_registry
SET human_label =
        'Name search: matches any recorded form of a person''s name -- '
        'canonical, alternate, misspelt, or Indigenous. Use this to find a person by name.',
    aliases = ARRAY[
        'name', 'any name', 'called', 'known as', 'goes by', 'find by name',
        'search by name', 'spelled', 'spelling', 'alternate name', 'other name',
        'indigenous name', 'native name', 'misspelling', 'variant name'
    ]
WHERE semantic_field = 'name_search';

-- Narrow student_name's aliases to what it actually is: the name as displayed. It kept
-- the bare alias "name", which is why it won every name lookup by default.
UPDATE assistant.field_registry
SET human_label = 'Name as recorded in the display cell',
    aliases = ARRAY['display name', 'full name', 'recorded name', 'student']
WHERE semantic_field = 'student_name';
