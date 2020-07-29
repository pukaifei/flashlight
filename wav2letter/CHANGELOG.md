# Changelog
All notable changes to wav2letter++ should be documented in this file. Use `Added` / `Changed` / `Removed` to appropriately mention your modifications.

#### 2019-01-02 : D13570032 (Vineel Pratap)
- **Changed** : Rename the class `ConnectionistTemporalCriterion` to `ConnectionistTemporalClassificationCriterion`

#### 2018-12-24 : D13545961 (Vineel Pratap)
- **Changed** : Use FLAGS_tokensdir as basepath while loading Tokens file in Test/Decode
- **Changed** : Use FLAGS_target default value to 'tkn'

#### 2018-12-20 : D13514973 (Vineel Pratap)
- **Changed** : Serialization of ASG

#### 2018-12-17 : D13487182 (Vineel Pratap)
- **Added** :  Parsing for PReLU layer `PR <OPTIONAL: numElements> <OPTIONAL: initValue>`
