from transformers import BertTokenizer
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
import numpy as np

import sys
sys.path.append('.')
import os
from news_data.crawlNews.crawlNewPaper import crawl_News
import glob
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import sent_tokenize
import re

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
        
    # def get_data_url(self, paper_url):
    #     title, description,paras = crawl_News(url=paper_url)
    #     return title, description, paras
            
    # def vector_calculator_url(self, paper_url):
    #     title, description, paras = self.get_data_url(paper_url)
    #     # centroid vector
    #     # Inputs are provided through numpy array
    #     input_id_title = self.tokenizer(title, return_tensors="pt")
    #     inputs_title_onnx = {k: v.cpu().detach().numpy() for k, v in input_id_title.items()}
    #     # Run the model (None = get all the outputs)
    #     _, title_pooled = self.cpu_model.run(None, inputs_title_onnx)
        
    #     # Inputs are provided through numpy array
    #     input_id_description = self.tokenizer(description, return_tensors="pt")
    #     inputs_description_onnx = {k: v.cpu().detach().numpy() for k, v in input_id_description.items()}
    #     # Run the model (None = get all the outputs)
    #     _, description_pooled = self.cpu_model.run(None, inputs_description_onnx)
        
    #     t_d = np.stack((title_pooled, description_pooled))
    #     centroid_doc = np.mean(t_d, axis=0)
        
    #     #vector sentences
    #     n = len(paras)
    #     sents_vec_dict = {v: k for v, k in enumerate(paras)}    
    #     for index in range(n) : 
    #         input_id = self.tokenizer(paras[index], return_tensors="pt")
    #         inputs_onnx = {k: v.cpu().detach().numpy() for k, v in input_id.items()}
    #         # Run the model (None = get all the outputs)
    #         _, pooled = self.cpu_model.run(None, inputs_onnx)
    #         sents_vec_dict.update({index:pooled})
            
    #     return sents_vec_dict, centroid_doc

    # def summary_url(self,paper_url, k):
    #     paras = self.get_data_url(paper_url)[2]
    #     sents_vec_dict, centroid_doc = self.vector_calculator_url(paper_url)
    #     cosine_sim = {}
    #     for key in sents_vec_dict.keys():
    #         cosine_2vec = cosine_similarity(centroid_doc, sents_vec_dict[key])
    #         cosine_sim.update({key:cosine_2vec})
    #     final_sim = sorted(cosine_sim.items(), key=lambda x:x[1], reverse=True)
    #     chossen = round(k*len(final_sim))
    #     list_index = dict(final_sim[:chossen]).keys()
    #     # print(list_index)
    #     result = []
    #     for index in sorted(list_index):
    #         result.append(paras[index])
    #     return '\n\n'.join(result)
    
    # def vector_calculator_doc(self, doc):
    #     doc = doc.replace('。','. ')
    #     doc = doc.replace('？','? ')
    #     doc = sent_tokenize(doc)
    #     # print(doc)
    #     n = len(doc)
    #     sents_vec_dict = {v: k for v, k in enumerate(doc)}    
    #     for index in range(n) : 
    #         input_id = self.tokenizer(doc[index], return_tensors="pt")
    #         inputs_onnx = {k: v.cpu().detach().numpy() for k, v in input_id.items()}
    #         # Run the model (None = get all the outputs)
    #         _, pooled = self.cpu_model.run(None, inputs_onnx)
    #         sents_vec_dict.update({index:pooled})
            
    #     X = list(sents_vec_dict.values())
    #     X = np.stack(X)
    #     # print(X.shape)
    #     centroid_doc = np.mean(X,0)  
    #     return sents_vec_dict, centroid_doc
    # def summary_doc(self, doc,k):
    #     # if auto_select_sent == False :
    #     #     k = 5
    #     # else :
    #     #     k = round(len(doc.split('.'))/ 2 + 1)
    #     doc = doc.replace('。','. ')
    #     doc = doc.replace('？','? ')
    #     doc_sents = sent_tokenize(doc)
    #     sents_vec_dict, centroid_doc = self.vector_calculator_doc(doc)
    #     cosine_sim = {}
    #     for key in sents_vec_dict.keys():
    #         cosine_2vec = cosine_similarity(centroid_doc, sents_vec_dict[key])
    #         # print(centroid_doc.shape, sents_vec_dict[key].shape)
    #         cosine_sim.update({key:cosine_2vec})
    #     final_sim = sorted(cosine_sim.items(), key=lambda x:x[1], reverse=True)
    #     chossen = round(k*len(final_sim))
        
    #     list_index = dict(final_sim[:chossen]).keys()
    #     result = []
        
    #     for index in sorted(list_index):
    #         result.append(doc_sents[index])
    #     mingg = '\n\n'.join(result)
    #     return mingg
    def sum_main(self, news, k):
        """
        news : dictionary bao gom :
            title : title cua news
            description : None(gan bang title) or string, description cua news
            paras :  doan tin cua news
        k : % van ban tom tat
        """
        title = news['title']
        description = news['description']
        paras = news['paras']
        if self.lang == 'ch':
            paras = [s.replace('\n','').replace('\u3000',' ').strip() for s in re.split('[。！？；]', doc)]
        else :
            paras = sent_tokenize(paras)
            paras = [s.replace('\n','').strip() for s in paras]
        print(paras)       
        if title.strip() != "" and description.strip() != "":
            input_id_title = self.tokenizer(title, return_tensors="pt")
            inputs_title_onnx = {k: v.cpu().detach().numpy() for k, v in input_id_title.items()}
            # Run the model (None = get all the return sents_vec_dict, centroid_docoutputs)
            _, title_pooled = self.cpu_model.run(None, inputs_title_onnx)
            
            # Inputs are provided through numpy array
            input_id_description = self.tokenizer(description, return_tensors="pt")
            inputs_description_onnx = {k: v.cpu().detach().numpy() for k, v in input_id_description.items()}
            # Run the model (None = get all the outputs)
            _, description_pooled = self.cpu_model.run(None, inputs_description_onnx)
            
            t_d = np.stack((title_pooled, description_pooled))
            centroid_doc = np.mean(t_d, axis=0)
            
            #vector sentences
            n = len(paras)
            sents_vec_dict = {v: k for v, k in enumerate(paras)}    
            for index in range(n) : 
                input_id = self.tokenizer(paras[index], return_tensors="pt")
                inputs_onnx = {k: v.cpu().detach().numpy() for k, v in input_id.items()}
                # Run the model (None = get all the outputs)
                _, pooled = self.cpu_model.run(None, inputs_onnx)
                sents_vec_dict.update({index:pooled})
                
            cosine_sim = {}
            for key in sents_vec_dict.keys():
                cosine_2vec = cosine_similarity(centroid_doc, sents_vec_dict[key])
                cosine_sim.update({key:cosine_2vec})
            final_sim = sorted(cosine_sim.items(), key=lambda x:x[1], reverse=True)
            chossen = round(k*len(final_sim))
            list_index = dict(final_sim[:chossen]).keys()
            # print(list_index)
            result = []
            for index in sorted(list_index):
                result.append(paras[index])
            return '\n\n'.join(result)
        elif title.strip() != "" and description.strip() == "":
            input_id_title = self.tokenizer(title, return_tensors="pt")
            inputs_title_onnx = {k: v.cpu().detach().numpy() for k, v in input_id_title.items()}
            # Run the model (None = get all the return sents_vec_dict, centroid_docoutputs)
            _, title_pooled = self.cpu_model.run(None, inputs_title_onnx)
            centroid_doc = title_pooled
            #vector sentences
            n = len(paras)
            sents_vec_dict = {v: k for v, k in enumerate(paras)}    
            for index in range(n) : 
                input_id = self.tokenizer(paras[index], return_tensors="pt")
                inputs_onnx = {k: v.cpu().detach().numpy() for k, v in input_id.items()}
                # Run the model (None = get all the outputs)
                _, pooled = self.cpu_model.run(None, inputs_onnx)
                sents_vec_dict.update({index:pooled})
                
            cosine_sim = {}
            for key in sents_vec_dict.keys():
                cosine_2vec = cosine_similarity(centroid_doc, sents_vec_dict[key])
                cosine_sim.update({key:cosine_2vec})
            final_sim = sorted(cosine_sim.items(), key=lambda x:x[1], reverse=True)
            chossen = round(k*len(final_sim))
            list_index = dict(final_sim[:chossen]).keys()
            # print(list_index)
            result = []
            for index in sorted(list_index):
                result.append(paras[index])
            return '\n\n'.join(result)
        elif title.strip() == "" and description.strip() == "":
            n = len(paras)
            sents_vec_dict = {v: k for v, k in enumerate(paras)}    
            for index in range(n) : 
                input_id = self.tokenizer(paras[index], return_tensors="pt")
                inputs_onnx = {k: v.cpu().detach().numpy() for k, v in input_id.items()}
                # Run the model (None = get all the outputs)
                _, pooled = self.cpu_model.run(None, inputs_onnx)
                sents_vec_dict.update({index:pooled})
            X = list(sents_vec_dict.values())
            X = np.stack(X)
            centroid_doc = np.mean(X,0)  
            cosine_sim = {}
            for key in sents_vec_dict.keys():
                cosine_2vec = cosine_similarity(centroid_doc, sents_vec_dict[key])
                cosine_sim.update({key:cosine_2vec})
            final_sim = sorted(cosine_sim.items(), key=lambda x:x[1], reverse=True)
            chossen = round(k*len(final_sim))
            list_index = dict(final_sim[:chossen]).keys()
            # print(list_index)
            result = []
            for index in sorted(list_index):
                result.append(paras[index])
            return '\n\n'.join(result)


if __name__ == '__main__':    
    vi_summ = M_Sum()
    doc = """Khi được hỏi về nguy cơ tiềm ẩn đối với khu vực phía nam của Ukraine trước cuộc tấn công quy mô lớn sắp xảy ra của Nga, Bộ trưởng Quốc phòng Ukraine Oleksii Reznikov hôm 12/2 cho biết Ukraine tìm cách ngăn Nga kiểm soát Biển Đen - vùng biển chiến lược trong chiến dịch quân sự của Nga ở Ukraine.

"Tôi thực sự không thích đưa ra dự đoán hay đánh giá ý kiến, nhưng để kiểm soát Odessa và khu vực (phía nam) nói chung, Nga phải chiếm ưu thế trên Biển Đen. Tuy nhiên, chúng tôi đã tước đi cơ hội này của họ", Bộ trưởng Reznikov nói trong một cuộc họp báo.

Odessa là thành phố đông dân thứ 3 của Ukraine và là một trung tâm du lịch, thương mại lớn nằm trên bờ Tây Bắc Biển Đen. Odessa cũng là điểm trung chuyển lớn với 3 thương cảng, đồng thời là ngã ba đường sắt lớn nhất phía Nam Ukraine, do đó Odessa có ý nghĩa quan trọng chiến lược không chỉ về thương mại mà cả quy hoạch quân sự. Odessa cũng là nơi đặt Bộ Tư lệnh Hải quân của quân đội Ukraine."""
    news = {'title':'',
            'description':'',
            'paras':doc}
    s1 = vi_summ.sum_main(news,0.4)
    print(s1)
    print('----------------------------------------------------------------')
    ###############################
    en_summ = M_Sum(lang='en')
    doc = """
    Yevgeny Prigozhin, the combative boss of Russia’s Wagner private military group, relishes his role as an anti-establishment maverick, but signs are growing that the Moscow establishment now has him pinned down and gasping for breath.

Prigozhin placed a bet on his mercenaries raising the Russian flag in the eastern Ukrainian city of Bakhmut, albeit at a considerable cost to the ranks of his force and probably to his own fortune.

He spent heavily on recruiting as many as 40,000 prisoners to throw into the fight, but after months of grinding battle and staggering losses he is struggling to replenish Wagner’s ranks, all the while accusing Russia’s Ministry of Defense of trying to strangle his force.

Many analysts think his suspicions are well-founded – that Russia’s military establishment is using the Bakhmut “meat-grinder” to cut him down to size or eliminate him as a political force altogether.

At the weekend, Prigozhin acknowledged that the battle in Bakhmut was “difficult, very difficult, with the enemy fighting for each meter.”

In another video message, Prigozhin said: “We need the military to shield the approaches (to Bakhmut). If they manage to do so, everything will be okay. If not, then Wagner will be encircled together with the Ukrainians inside Bakhmut.”
    """
    news = {'title':'',
            'description':'',
            'paras':doc}
    s2 = en_summ.sum_main(news,0.4)
    print(s2)
    print('----------------------------------------------------------------')
    
    ##################################
    ru_summ = M_Sum(lang='ru')
    doc = """
    В суде Лерчек и ее супруг просили суд не лишать из интернета, поскольку на нем завязан весь их бизнес - помимо самого блога это еще и фирма по производству косметики. Чекалин объявил, что у них официально трудоустроены 250 человек.

Но судья согласилась с доводами следствия: с помощью интернета, мобильной и телефонной связи супруги могут оказывать влияние на свидетелей.

Валерия уехала домой сразу после заседания, а Артему (решение по нему выносилось отдельно) пришлось задержаться.
    """
    news = {'title':'',
            'description':'',
            'paras':doc}
    s3 = ru_summ.sum_main(news,0.4)
    print(s3)
    print('----------------------------------------------------------------')
    
    #########################################
    ch_summ = M_Sum(lang='ch')
    doc = """
    　新华社北京3月15日电（记者彭韵佳、沐铁城）为加强医疗保障基金监督检查，规范飞行检查工作，国家医保局近日印发《医疗保障基金飞行检查管理暂行办法》，自2023年5月1日起施行。

　　办法明确，有下列情形之一的，医疗保障行政部门可以启动飞行检查，包括年度工作计划安排的；举报线索反映医疗保障基金可能存在重大安全风险的；医疗保障智能监控或者大数据筛查提示医疗保障基金可能存在重大安全风险的；新闻媒体曝光，造成重大社会影响的；其他需要开展飞行检查的情形。

　　办法提出，被检地医疗保障行政部门应当在收到移交材料的30个工作日内，将处理进度和整改方案上报组织飞行检查的医疗保障行政部门，并在处理完结后5个工作日内报送书面报告。此外，组织飞行检查的医疗保障行政部门应当及时将典型案例向社会公告。
    """
    news = {'title':'',
            'description':'',
            'paras':doc}
    s4 = ch_summ.sum_main(news,0.4)
    print(s4)