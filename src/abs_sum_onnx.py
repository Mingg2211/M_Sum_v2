# import torch
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from tokenizer import Tokenizer

class abs_Sum():
    def __init__(self,k, lang='vietnamese'):
        self.lang = lang
        self.k = k
    def summary(self, doc):
        parser = PlaintextParser.from_string(doc, Tokenizer(self.lang))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, self.k) 
        summary = [str(sentence) for sentence in summary]
        return '\n\n'.join((summary))


doc = """
Khi được hỏi về nguy cơ tiềm ẩn đối với khu vực phía nam của Ukraine trước cuộc tấn công quy mô lớn sắp xảy ra của Nga, Bộ trưởng Quốc phòng Ukraine Oleksii Reznikov hôm 12/2 cho biết Ukraine tìm cách ngăn Nga kiểm soát Biển Đen - vùng biển chiến lược trong chiến dịch quân sự của Nga ở Ukraine.

"Tôi thực sự không thích đưa ra dự đoán hay đánh giá ý kiến, nhưng để kiểm soát Odessa và khu vực (phía nam) nói chung, Nga phải chiếm ưu thế trên Biển Đen. Tuy nhiên, chúng tôi đã tước đi cơ hội này của họ", Bộ trưởng Reznikov nói trong một cuộc họp báo.

Odessa là thành phố đông dân thứ 3 của Ukraine và là một trung tâm du lịch, thương mại lớn nằm trên bờ Tây Bắc Biển Đen. Odessa cũng là điểm trung chuyển lớn với 3 thương cảng, đồng thời là ngã ba đường sắt lớn nhất phía Nam Ukraine, do đó Odessa có ý nghĩa quan trọng chiến lược không chỉ về thương mại mà cả quy hoạch quân sự. Odessa cũng là nơi đặt Bộ Tư lệnh Hải quân của quân đội Ukraine.

Ông Reznikov đề cập đến việc Ukraine từng sử dụng Neptune, vũ khí chống hạm được sản xuất ở Ukraine, để nhắm vào tàu tuần dương Moskva của Nga hồi năm ngoái.

"Chúng tôi đã ngăn chặn sự thống trị của Nga ở Biển Đen, đặc biệt sau khi phóng thành công tên lửa Neptune khiến tàu tuần dương Moskva bị chìm tại khu vực này", ông Reznikov nói.
"""     
v_s = abs_Sum(lang='vietnamese', k=2)
print(v_s.summary(doc))

doc = """
    　新华社北京3月15日电（记者彭韵佳、沐铁城）为加强医疗保障基金监督检查，规范飞行检查工作，国家医保局近日印发《医疗保障基金飞行检查管理暂行办法》，自2023年5月1日起施行。

　　办法明确，有下列情形之一的，医疗保障行政部门可以启动飞行检查，包括年度工作计划安排的；举报线索反映医疗保障基金可能存在重大安全风险的；医疗保障智能监控或者大数据筛查提示医疗保障基金可能存在重大安全风险的；新闻媒体曝光，造成重大社会影响的；其他需要开展飞行检查的情形。

　　办法提出，被检地医疗保障行政部门应当在收到移交材料的30个工作日内，将处理进度和整改方案上报组织飞行检查的医疗保障行政部门，并在处理完结后5个工作日内报送书面报告。此外，组织飞行检查的医疗保障行政部门应当及时将典型案例向社会公告。
    """
ch_s = abs_Sum(lang='chinese', k=2)
print(ch_s.summary(doc))

doc = """
    В суде Лерчек и ее супруг просили суд не лишать из интернета, поскольку на нем завязан весь их бизнес - помимо самого блога это еще и фирма по производству косметики. Чекалин объявил, что у них официально трудоустроены 250 человек.

Но судья согласилась с доводами следствия: с помощью интернета, мобильной и телефонной связи супруги могут оказывать влияние на свидетелей.

Валерия уехала домой сразу после заседания, а Артему (решение по нему выносилось отдельно) пришлось задержаться.
    """
ru_s = abs_Sum(lang='russian', k =2)
print(ru_s.summary(doc))

doc = """
    Yevgeny Prigozhin, the combative boss of Russia’s Wagner private military group, relishes his role as an anti-establishment maverick, but signs are growing that the Moscow establishment now has him pinned down and gasping for breath.

Prigozhin placed a bet on his mercenaries raising the Russian flag in the eastern Ukrainian city of Bakhmut, albeit at a considerable cost to the ranks of his force and probably to his own fortune.

He spent heavily on recruiting as many as 40,000 prisoners to throw into the fight, but after months of grinding battle and staggering losses he is struggling to replenish Wagner’s ranks, all the while accusing Russia’s Ministry of Defense of trying to strangle his force.

Many analysts think his suspicions are well-founded – that Russia’s military establishment is using the Bakhmut “meat-grinder” to cut him down to size or eliminate him as a political force altogether.

At the weekend, Prigozhin acknowledged that the battle in Bakhmut was “difficult, very difficult, with the enemy fighting for each meter.”

In another video message, Prigozhin said: “We need the military to shield the approaches (to Bakhmut). If they manage to do so, everything will be okay. If not, then Wagner will be encircled together with the Ukrainians inside Bakhmut.”
    """
en_s = abs_Sum(lang='english', k=2)
print(en_s.summary(doc))