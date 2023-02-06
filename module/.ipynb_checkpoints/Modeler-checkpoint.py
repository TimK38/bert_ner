from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences #2.2.4
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,Dataset# 
import transformers
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup,BertTokenizer 

from seqeval.metrics import f1_score, accuracy_score
import torch

import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from tqdm import tqdm, trange
import pandas as pd 

from torch.nn.utils.rnn import pad_sequence

class modeler:
    def __init__(self , bert_model , entity ):
        '''
              
        '''
        self.tokenizer_chinese = BertTokenizer.from_pretrained( bert_model, do_lower_case=False) # "ckip-base-chinese-ner"
        self.bert_model = bert_model 
        self.MAX_LEN = 75
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
        self.bs = 32 # batch sized
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenize_and_preserve_labels(self, sentence, text_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self.tokenizer_chinese.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels 
    def preprocess(self , df_train):
        sentences = df_train['words'].tolist()
        labels = df_train['tags_encoding'].tolist()
        tokenized_texts_and_labels = [  self.tokenize_and_preserve_labels(sent, labs)
                                        for sent, labs in zip(sentences, labels)   ]
        
        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
        labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
        
        input_ids = pad_sequences([self.tokenizer_chinese.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=self.MAX_LEN, dtype="long", value=0.0, #　MAX_LEN
                          truncating="post", padding="post")
        tags =      pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=self.MAX_LEN, value=self.tag2idx["PAD"], padding="post",
                         dtype="long", truncating="post")
        attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
        
        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=38, test_size=0.1)
        tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                            random_state=38, test_size=0.1)
        tr_inputs = torch.tensor(tr_inputs)
        tr_inputs = torch.tensor(tr_inputs).to(torch.int64)

        val_inputs = torch.tensor(val_inputs)
        val_inputs = torch.tensor(val_inputs).to(torch.int64)

        tr_tags = torch.tensor(tr_tags)
        tr_tags=torch.tensor(tr_tags, dtype=torch.long) 

        val_tags = torch.tensor(val_tags)
        val_tags=torch.tensor(val_tags, dtype=torch.long) 

        tr_masks = torch.tensor(tr_masks)
        val_masks = torch.tensor(val_masks)
        
        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.bs)#總資料數45081,batch_size:32,所以共有1409個batch

        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.bs)
        return train_dataloader , valid_dataloader
    def train(self , BertForTokenClassification_model , train_dataloader , valid_dataloader , save_model_path ):
        model = BertForTokenClassification.from_pretrained(
                                            BertForTokenClassification_model , #"ckip-base-chinese-ner"　
                                            num_labels=73,#len(tag2idx),
                                            output_attentions = False,
                                            output_hidden_states = False  )
        model.classifier = torch.nn.Linear(768, len(self.tag2idx))
        model.num_labels = len(self.tag2idx)
        model.cuda()
        
        ##參數設定，參數內容暫且寫死
        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=3e-5,
            eps=1e-8 )
        
        epochs = 3
        max_grad_norm = 1.0

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps)
        ## Store the average loss after each epoch so we can plot them.
        loss_values, validation_loss_values = [], []

        for _ in trange(epochs, desc="Epoch"):
            # ========================================
            #               Training
            # ========================================
            # Perform one full pass over the training set.

            # Put the model into training mode.
            model.train()
            # Reset the total loss for this epoch.
            total_loss = 0

            # Training loop
            for step, batch in enumerate(train_dataloader):
                # add batch to gpu
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # Always clear any previously calculated gradients before performing a backward pass.
                model.zero_grad()
                # forward pass
                # This will return the loss (rather than the model output)
                # because we have provided the `labels`.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
                # get the loss
                loss = outputs[0]
                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # track train loss
                total_loss += loss.item()
                # Clip the norm of the gradient
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
                # update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)
            print("Average train loss: {}".format(avg_train_loss))

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)


            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            # Put the model into evaluation mode
            model.eval()
            # Reset the validation loss for this epoch.
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions , true_labels = [], []
            for batch in valid_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                # Telling the model not to compute or store gradients,
                # saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have not provided labels.
                    outputs = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels)
                # Move logits and labels to CPU
                logits = outputs[1].detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                eval_loss += outputs[0].mean().item()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)

            eval_loss = eval_loss / len(valid_dataloader)
            validation_loss_values.append(eval_loss)
            print("Validation loss: {}".format(eval_loss))
            pred_tags = [self.tag_values[p_i] for p, l in zip(predictions, true_labels)
                                         for p_i, l_i in zip(p, l) if self.tag_values[l_i] != "PAD"]
            valid_tags = [self.tag_values[l_i] for l in true_labels
                                          for l_i in l if self.tag_values[l_i] != "PAD"]
            print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
            print("Validation F1-Score: {}".format(f1_score([pred_tags], [valid_tags])))
            print('*' * 100)
        # save model 
        model.save_pretrained( save_model_path ) #　'model/ner_model'
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        plt.plot(loss_values, 'b-o', label="training loss")
        plt.plot(validation_loss_values, 'r-o', label="validation loss")

        # Label the plot.
        plt.title("Learning curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()
        
        return model
    def predict_for_one(self ,text , save_model_path  ):
        '''受限模型關係只能預測前512字'''
        # 讀取模型
        model = BertForTokenClassification.from_pretrained(save_model_path,num_labels=len(self.tag2idx),
                                                                            output_attentions = False,
                                                                            output_hidden_states = False , local_files_only=True) 
        model.cuda()
        tokenized_sentence = self.tokenizer_chinese.encode(text[:512]) # 
        input_ids = torch.tensor([tokenized_sentence]).cuda()
        with torch.no_grad():
            output = model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        # join bpe split tokens
        tokens = self.tokenizer_chinese.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(self.tag_values[label_idx])
                new_tokens.append(token)
        output = ""
        for i in range(len(new_labels)):
            if new_labels[i]!='O':
                output += new_tokens[i] + '({type_})'.format(type_ = new_labels[i])
            else:
                output += new_tokens[i]
        return output

    def batch_predict(self, df_predict , save_model_path , batch_size ):
        #save_model_path = 'model/ner_model/'
        model = BertForTokenClassification.from_pretrained(save_model_path,num_labels=len(self.tag2idx),
                                                                            output_attentions = False,
                                                                            output_hidden_states = False , local_files_only=True) 
        model.cuda()
        
        news_texts = df_predict['words'].apply(lambda  x:x[:510]).tolist()
        news_id = df_predict.index.tolist()
        input_dataset = NER_Dataset((news_id,news_texts), self.bert_model , self.device ) #　"ckip-base-chinese-ner" 
        TestDataLoader = DataLoader( input_dataset,
                                batch_size = batch_size,
                                shuffle = False,
                                num_workers = 0,
                                pin_memory = False,
                                collate_fn = input_dataset.collate_fn)
        
        news_idx,sentence_idx,predict_idx,sentence = [],[],[],[]
        for idx,batch  in tqdm(enumerate(TestDataLoader),total = len(TestDataLoader)):

            batch_idx,batch_sentence_idx,batch_sentence = batch 
            batch_sentence_idx = batch_sentence_idx.to(self.device)
            with torch.no_grad():
                predict_result = model(batch_sentence_idx)
            predict_result = torch.argmax(predict_result[0], dim=2)

            news_idx.extend(batch_idx)
            predict_idx.extend([predict_result[i,:].to('cpu').tolist() for i in range(predict_result.shape[0])])
            sentence_idx.extend([batch_sentence_idx[i,:].to('cpu').tolist() for i in range(batch_sentence_idx.shape[0])])
            sentence.extend(batch_sentence)
        output = pd.DataFrame( data = {'sentence':sentence ,'predict': predict_idx} )
        output['predict'] = output['predict'].apply(lambda x :[ self.tag_values[i] for i in x ][1:-1] )#掐頭去尾
        return output

class NER_Dataset(Dataset):
    def __init__(self, input_data ,bert_token_path, device):
        self.idx, self.sentence = input_data
        self.tokenizer = BertTokenizer.from_pretrained(bert_token_path, do_lower_case=False) # "ckip-base-chinese-ner" 
        self.device = device # torch.device("cuda" if torch.cuda.is_available() else "cpu")# 


        assert len(self.idx) == len(self.sentence)
        self.sentence_idx = self.convert_to_idx(self.sentence)

    def convert_to_idx(self,sentence): 
        sentence_idx = []
        for s in sentence:      
            #sentence_idx.append([101] + self.tokenizer.convert_tokens_to_ids(list(s.lower())) + [102])     
            sentence_idx.append([101] + self.tokenizer.convert_tokens_to_ids(list(s)) + [102])        
        assert len(sentence) == len(sentence_idx)
        return sentence_idx
    
    def __len__(self):
        return len(self.sentence)

    def __getitem__(self,index):
        return self.idx[index], self.sentence_idx[index],self.sentence[index]

    def collate_fn(self,batch):
        batch_idx = [row[0] for row in batch]
        batch_sentence_idx = [torch.tensor(row[1]) for row in batch]
        batch_sentence = [row[2] for row in batch]

        batch_sentence_idx = pad_sequence(batch_sentence_idx,batch_first = True)

        return batch_idx,batch_sentence_idx,batch_sentence