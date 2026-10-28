-- Alias corrections found by running the question suite.
--
-- "died at" was registered as an alias of LOCATION OF DEATH, so "died at each age
-- between 7 and 19" resolved to a place field instead of the age field. Aliases that
-- are sentence fragments rather than names of the thing being stored match too
-- eagerly; the remaining aliases all name the field itself.

UPDATE assistant.field_registry SET aliases = ARRAY[
    'death location', 'location of death', 'death setting', 'where they died',
    'died at the school', 'place category'
] WHERE file_id = 91 AND semantic_field = 'location_of_death';

UPDATE assistant.field_registry SET aliases = ARRAY[
    'age', 'death age', 'age at death', 'ages at death', 'years old',
    'youngest', 'oldest', 'age at time of death'
] WHERE file_id = 91 AND semantic_field = 'age_at_death';

UPDATE assistant.field_registry SET aliases = ARRAY[
    'death date', 'date of death', 'died', 'dod', 'death year', 'year of death',
    'deaths', 'student deaths', 'confirmed deaths'
] WHERE file_id = 91 AND semantic_field = 'death_date';

UPDATE assistant.field_registry SET aliases = ARRAY[
    'birth date', 'date of birth', 'born', 'dob', 'birth'
] WHERE file_id IN (49, 91) AND semantic_field = 'birth_date';
