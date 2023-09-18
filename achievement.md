## 20230918
针对之前的文章*Learning to Retrieve In-Context Examples for Large Language Models*，找到了前序文章*UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation*，明确了其中几个实现细节，包括检索器的输入输出、prompt评分的方式等。

值得注意的是本文对prompt design、prompt tuning与prompt retrieval作了明确的区分。prompt design一般指使用自然语言提示模型如何完成任务，prompt tuning一般指使用soft prompt（通过hard prompt方式初始化，即用人工可阅读的单词序列初始化，然后通过向量空间连续优化得到新的提示）调优模型性能，prompt retrieval一般外挂一个检索器，使用检索器检索自然语言组成的prompt，既有灵活性也有可解释性，大模型用于评估检索器检索出的prompt的性能。
