from datasets import load_dataset, load_metric,ClassLabel, Sequence
import random
import itertools
import pandas as pd 
import numpy as np 
from opencc import OpenCC

class load_data:
    def __init__(self ):
        self.cc = OpenCC('s2tw')
        self.entity_dict = { 0: 'O',
                             1: 'B-PERSON',
                             2: 'I-PERSON',
                             3: 'B-NORP',
                             4: 'I-NORP',
                             5: 'B-FAC',
                             6: 'I-FAC',
                             7: 'B-ORG',
                             8: 'I-ORG',
                             9: 'B-GPE',
                             10: 'I-GPE',
                             11: 'B-LOC',
                             12: 'I-LOC',
                             13: 'B-PRODUCT',
                             14: 'I-PRODUCT',
                             15: 'B-DATE',
                             16: 'I-DATE',
                             17: 'B-TIME',
                             18: 'I-TIME',
                             19: 'B-PERCENT',
                             20: 'I-PERCENT',
                             21: 'B-MONEY',
                             22: 'I-MONEY',
                             23: 'B-QUANTITY',
                             24: 'I-QUANTITY',
                             25: 'B-ORDINAL',
                             26: 'I-ORDINAL',
                             27: 'B-CARDINAL',
                             28: 'I-CARDINAL',
                             29: 'B-EVENT',
                             30: 'I-EVENT',
                             31: 'B-WORK_OF_ART',
                             32: 'I-WORK_OF_ART',
                             33: 'B-LAW',
                             34: 'I-LAW',
                             35: 'B-LANGUAGE',
                             36: 'I-LANGUAGE'}
        
    def setence_combine(self, a , cols ):
        return list(itertools.chain.from_iterable([i[cols] for i in a ]))
    
    def Hugging_face_df_2_pandas(self, dataset ):
        df = pd.DataFrame(dataset)#[picks])
        for column, typ in dataset.features.items():
            if isinstance(typ, ClassLabel):
                df[column] = df[column].transform(lambda i: typ.names[i])
            elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
                df[column] = df[column].transform(lambda x: [typ.feature.names[i][0] for i in x])
        df['sentences_comb'] = df['sentences'].apply(lambda x : self.setence_combine(x , 'words'))
        df['sentences_comb_label'] = df['sentences'].apply(lambda x : self.setence_combine(x , 'named_entities'))
        return df  
    
    def process( self, sentences_comb, sentences_comb_label ):
        sentence_split = []
        label_split = []
        dict_ = self.entity_dict
        for i in range(len(sentences_comb)):

            if sentences_comb_label[i] in [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35]:

                sentence_split.append( [i for i in sentences_comb[i]]  )   

                label_split_sub = [  dict_[sentences_comb_label[i]]  ] + [ dict_[sentences_comb_label[i] + 1] ] * (len(sentences_comb[i])-1)
                label_split.append( label_split_sub )  
            else:
                sentence_split.append( [i for i in sentences_comb[i]]  )   
                label_split_sub = [ dict_[ sentences_comb_label[i] ]] * (len(sentences_comb[i]))
                label_split.append( label_split_sub )  

        sentence_split = list(itertools.chain.from_iterable(sentence_split))
        label_split = list(itertools.chain.from_iterable(label_split))
        return [ sentence_split,label_split ]
    
    def process_tags_encoding(self,  words_tags_encoding ):
        # 把tags 從 [ [words] , [tags ] ]中的1取出，並且新增End flag到每個標籤最後一部分  
        encoding = words_tags_encoding[1]
        result = []
        for i in range(len(encoding)):
            try:
                if encoding[i][0]=='I' and  encoding[i+1]=='O':
                    result.append( 'E' + encoding[i][1:] )
                else:
                    result.append( encoding[i] )
            except:
                result.append( encoding[i] )
        return result
    
    def process_words(self , words_tags_encoding ):
        # 把words 從 [ [words] , [tags ] ]中的０取出  
        encoding = words_tags_encoding[0]
        return encoding
    
    def strQ2B(self , ustring):# 處理全行半行
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 12288:                            # 全形空格直接轉換
                inside_code = 32
            elif 65281 <= inside_code <= 65374:   				# 全形字元（除空格）根據關係轉化
                inside_code -= 65248
            rstring += chr(inside_code)
        return rstring

    def words_transform(self, text):
        cc = self.cc 
        text = self.strQ2B(text)
        text = cc.convert(text)
        return text

    def Load_Data(self , dataset , data_type  ):
        dataset = load_dataset(dataset , 'chinese_v4', split = data_type )
        df = self.Hugging_face_df_2_pandas(dataset)        
        df['words_&_tags_encoding'] = df.apply(lambda x : self.process(x.sentences_comb , x.sentences_comb_label),axis = 1 ) 
        df['tags_encoding'] = df['words_&_tags_encoding'].apply(lambda x : self.process_tags_encoding(x)    )
        df['words'] = df['words_&_tags_encoding'].apply(lambda x : self.process_words(x)    )
        df = df[['tags_encoding','words']]#.head()
        df['words'] = df['words'].apply(lambda x :[self.words_transform(i) for i in x ])
        return df

class decide_entity_label:
    def __init__(self , entity):
        '''
        Entity 種類:PERSON、NORP、FAC、ORG、GPE、LOC、PRODUCT、DATE、TIME、PERCENT、MONEY、QUANTITY、ORDINAL、CARDINAL、EVENT、WORK_OF_ART、LAW、LANGUAGE        
        '''
        self.tag_values = ['O',
         'B-{EVENT}'.format( EVENT = entity ),
         'I-{EVENT}'.format( EVENT = entity ),
         'E-{EVENT}'.format( EVENT = entity ),
         'PAD']
        self.tag2idx = { 'O': 0,
                   'B-{EVENT}'.format( EVENT = entity ) : 1,
                   'I-{EVENT}'.format( EVENT = entity ) : 2,
                   'E-{EVENT}'.format( EVENT = entity ) : 3,
                   'PAD': 4}
        
    def split_eve(self , event , word):
        # 根據決定的類型，先會去掉其他Label，同時怕訓練資料變得過於稀疏，把資料進行重組
        split_event = []
        split_word = []
        for i in range(len(event)):
            try:
                if event[i]=='O' and event[i+1]!='O':
                    n = random.randint(0,75)
                    start_index = i-n
                    end_index = i-n+75
                    if start_index<0:
                        split_event.append(event[:75])
                        split_word.append(word[:75])
                    elif end_index > len(event):
                        split_event.append(event[-75:])
                        split_word.append(word[-75:])
                    else:
                        split_event.append(event[start_index:end_index])
                        split_word.append(word[start_index:end_index])
                else:
                    continue
            except:
                x = 0
        return [split_event , split_word]
    
    def Decide_Entity_Label( self , df ):
        #### 資料重新切割
        df['tags_encoding'] = df['tags_encoding'].apply(lambda x : [i if i in self.tag_values else "O" for i in x   ] ) 
        df['split'] = df.apply(lambda x : self.split_eve(x.tags_encoding , x.words) ,axis =1 )
        new_words = []
        new_tags_encoding = []
        for i in range(len(df)):
            spl_word = df.iloc[i]['split'][1]
            spl_tag_encoding = df.iloc[i]['split'][0]
            for j in range(len(spl_word)):
                new_words.append( spl_word[j] )
                new_tags_encoding.append( spl_tag_encoding[j] )
        return pd.DataFrame( data = { 'tags_encoding':new_tags_encoding,'words':new_words } )