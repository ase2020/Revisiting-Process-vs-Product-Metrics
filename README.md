# Revisiting Process versus Product Metrics: a Large Scale Analysis
Numerous automated SE methods can build predictive models from software project data. But what methods and conclusions should we endorse as we move from analytics in-the small (dealing with ahandful or projects) to analytics in-the-large (dealing with hundredsof projects).? To answer this question, we recheck prior small scale results (about process versus product metrics for defect prediction)using 722,471 commits from 770 Github projects We find that some analytics in-the-small conclusions still hold when scaling up to analytics in-the large. For example, like priorwork, we see that process metrics are better predictors for defects than product metrics (best process/product-based learners respec-tively achieve recalls of 98%/44% and AUCs of 95%/54%, median values). However, we warn that it is unwise to trust metric importance results from analytics in-the-small studies since those change, dramatically when moving to analytics in-the-large. Also, when reasoning in-the-large about hundreds of projects, it is better to use predictions from multiple models (since single model prediction scan become very confused and exhibit very high variance). Apart from the above specific conclusions, our more generalpoint is that the SE community now needs to revisit many of theconclusions previously obtained via analytics in-the-small.

## This Repository contains the code need to run the RQs of the research paper above. To run the code follow the instruction - 

1) Unzip src.tar.gz, data.tar.gz and results.tar.gz.
2) Run Pre-requisite.sh to generate all the necessary files to compile the results.
3) Run RQ1.sh to generate the results for RQ1. The results will be stored inside ''results/image''.
4) Run RQ2.sh to generate the results for RQ2. The results will be stored inside ''results/image''.
5) Run RQ3.sh to generate the results for RQ3. The results will be stored inside ''results/image''.
