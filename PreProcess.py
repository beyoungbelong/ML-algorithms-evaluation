import json
import pandas as pd
from nltk.corpus import stopwords
from string import punctuation
from zhon.hanzi import punctuation as pcn
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from tqdm import tqdm

wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    # 返回单词的词性,用来词性还原
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                    'VB': 'v', 'RB': 'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

def lemmatize_sent(text):
    # 输入原始数据，分词后获取词性。遍历分词后的pos_tag(word_tokenize(text))，返回词性还原后的小写单词列表
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=
    
    penn2morphy(tag))
            for word, tag in pos_tag([word.strip("'") for word in word_tokenize(text)
                                        if len(word.strip("'")) != 0])]


stopwords_json = {
			"en": ["a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after",
			       "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along",
			       "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any",
			       "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear",
			       "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking", "associated",
			       "at", "available", "away", "awfully", "b", "be", "became", "because", "become", "becomes",
			       "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside",
			       "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's",
			       "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes",
			       "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering",
			       "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "currently",
			       "d", "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't",
			       "doing", "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either",
			       "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every",
			       "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f",
			       "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former",
			       "formerly", "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting",
			       "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't",
			       "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help",
			       "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
			       "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd",
			       "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed",
			       "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isn't",
			       "it", "it'd", "it'll", "it's", "its", "itself", "j", "just", "k", "keep", "keeps", "kept", "know",
			       "known", "knows", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest",
			       "let", "let's", "like", "liked", "likely", "little", "look", "looking", "looks", "ltd", "m",
			       "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover",
			       "most", "mostly", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly",
			       "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no",
			       "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere",
			       "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones",
			       "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out",
			       "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed",
			       "please", "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r",
			       "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively",
			       "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly",
			       "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent",
			       "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six",
			       "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat",
			       "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup",
			       "sure", "t", "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx",
			       "that", "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there",
			       "there's", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they",
			       "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly",
			       "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took",
			       "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un", "under",
			       "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful",
			       "uses", "using", "usually", "uucp", "v", "value", "various", "very", "via", "viz", "vs", "w", "want",
			       "wants", "was", "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome", "well", "went",
			       "were", "weren't", "what", "what's", "whatever", "when", "whence", "whenever", "where", "where's",
			       "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
			       "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish",
			       "with", "within", "without", "won't", "wonder", "would", "wouldn't", "x", "y", "yes", "yet", "you",
			       "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "z", "zero"]}
stopwords_customize = set(['al', "n't", 'α', 'χ', 'δ'])  # 自定义的停用词
stopwords_json_en = set(stopwords_json['en'])  # 将网上Copy的停用词表stopwords_json转为set
stopwords_nltk_en = set(stopwords.words('english'))  # nltk里面的停用词表
stopwords_punct = set(punctuation + pcn)  # 增加了中文标点符号
stoplist_combined = set.union(stopwords_customize, stopwords_json_en, stopwords_nltk_en,
                                    stopwords_punct)  # 最终的停用词表

data_full_text = pd.read_csv('temp/abs.csv')
json_full_text = data_full_text.to_json(orient='records')
json_full_text = json.loads(json_full_text)

# 遍历论文，进行分词
count = 0
Num_None = 0
for paper in tqdm(json_full_text):
    count += 1
    if count % 1000 == 0:
        print(f"正在处理第{count}篇分词")
    content = paper['abstract']
    try:
        # 输入原始数据，遍历词性还原后的小写单词列表，将停用词和数字去掉，返回筛选后的结果。
        paper['tokenize'] = [word
                        for word in lemmatize_sent(content)
                        if word not in stoplist_combined
                        and not word.isdigit()
                        and '.' not in word
                        and '=' not in word
                        and "'" not in word and "\\u00" not in word
                        and len(word) >= 3]
    except:
        Num_None = Num_None + 1


with open('temp/分词后_abs.json', mode="w", encoding="utf-8") as f:
    json.dump(json_full_text, f, ensure_ascii=False)

