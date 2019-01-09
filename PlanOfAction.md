# Plan of action Complex Disorder stratification model

**Reminder**: 

AMIA Informatics Summit 2019 early registration due on the __2/21/2019__: 
* Student Non-Member 595$;

* Author Non-Member 1280$

Presentation: Run the new model on the same Multiple Myeloma data.

## Goal:
(1) __Quantitative approach:__

Focus on the methodology, find a metric to compare our method to the baselines.

- KDD 2019 conference: 

> August 3 - 7, 2019
Anchorage, Alaska USA.
**Call for papers deadline**: February 3rd, 2019

- SIGIR 2019 conference:

> July 21-25, 2019 
Paris, France.
**Call for full papers**: (Abstract) Mon, Jan 21, 2019; (submission deadline) Mon, Jan 28, 2019.
**Call for short papers**: (Abstract) Tue, Feb 12, 2019; (submission deadline) Tue, Feb 19, 2019

(2) __Quantitative/Qualitative approach:__

To submit to a Journal, investigate different diseases.

## (1) Quantitative approach steps

Paper outline:

> _Introduction_

> _Related Works_

> _Methods_

> _Evaluation: dataset; Silhouette; Clustering; Similarity(?)_

> _Disease stratification_

For the _evaluation_ step we can try:

- Similarity measure (Euclidean/Cosine) for each disorder with a table:

|model_name| Disease_A | Disease_B | Disease_C |  
| ---   | :-------: |:---------:| :--------:|   
| Model1| x         | x         | x         |
| Model2| x         | x         | x         |
| ...   | ...       | ...       | ...       |
| ModelN| x         | x         | x         |

Where $x$ is the chosen metric. We can average the similarity within patients with the same disorder between the different patient representations.
- Silhouette Index for number of clusters.

For the disease stratification step:

- Consider how many patients have a specific medical term among the most frequent (relatively frequent) ones.
- Chi-squared test of the frequencies of the one-hot encoded patient representations among unsupervised clusters. 

## To-dos
- Try adding a second layer to the CNN;
- Run the model on 7/8 distinct disorders + random patients. Dataset numerosity ~ 50,000/100,000 subjects.

Alzheimer [key word: alzheimer's disease (G30.9/331.0)] (N=4,905): most common cause of dementia. Greatest know risk factor is increasing age. The majority of people with Alzheimer's are > 65. SNOMED: alzheimer.
 
Multiple myeloma [key word: multiple myeloma (C90.00-02/203.00-02)] (N=4,727): bone marrow plasma cells cancer. Risk increases with age (> 60). SNOMED: multiple myeloma.

Parkinson's [key word: parkinson's disease (G20/332.0)](N=5,821): progressive nervous system disorder that affects movements. Risk increases with age (>= 60). SNOMED: parkinson

Breast cancer [key word: malignant neoplasm of female breast (C50.919/174.9 and 198.81/C79.81 = secondary malignant neoplasm of breast)] !!CHECK C50!! (N=11,822): risk increases with age. SNOMED: malignant neoplasm of female breast.

Prostate cancer [key word: malignant tumor of prostate (C61/185)] (N=11,106): risk increases with age. SNOMED: malignant tumor of prostate.

Hypertension.

Heart Disease.
