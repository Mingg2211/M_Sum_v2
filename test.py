# import libs
from sklearn.cluster import KMeans
from transformers import BertTokenizer
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import sent_tokenize
import os
class M_Sum():
    def create_model_for_provider(self,folder_model_path: str, provider: str) -> InferenceSession: 

        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        model_path=''
        # Load the model as a graph and prepare the CPU backend 
        for file in os.listdir(folder_model_path):
            if file.endswith(".onnx"):
                model_path=os.path.join(folder_model_path,file)
            
        if model_path=='':
            return print("Could found model")
        session = InferenceSession(model_path, options, providers=[provider])
        session.disable_fallback()
            
        return session
    
    def __init__(self,lang='vi'):
        self.lang = lang
        self.pretrained = "model/ViBert/vi_Bert_onnx" if self.lang=='vi' \
        else("model/ChBert/ch_Bert_onnx" if self.lang=='ch' \
        else ("model/RuBert/ru_Bert_onnx" if self.lang=='ru' \
        else 'model/EnBert/en_Bert_onnx'))
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained, local_files_only=True)
        self.cpu_model = self.create_model_for_provider(self.pretrained, "CPUExecutionProvider")
    
    def vector_calulator(self, doc:str):
        pass
    def Kmeans_summarize(self, num_cluster):
        pass