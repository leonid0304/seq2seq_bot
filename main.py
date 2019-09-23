#! /usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask
# tqdm визуализирует процесс работы цикла
from tqdm import tqdm
from sklearn.utils import shuffle
from data.twitter import data
from tensorlayer.models.seq2seq import Seq2seq
from tensorlayer.models.seq2seq_with_attention import Seq2seqLuongAttention
import os

# функция обработки данных
def initial_setup(data_corpus):
    metadata, idx_q, idx_a = data.load_data(PATH='data/{}/'.format(data_corpus))
    (trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)
    # удаляет отступы
    trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
    trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
    testX = tl.prepro.remove_pad_sequences(testX.tolist())
    testY = tl.prepro.remove_pad_sequences(testY.tolist())
    validX = tl.prepro.remove_pad_sequences(validX.tolist())
    validY = tl.prepro.remove_pad_sequences(validY.tolist())
    return metadata, trainX, trainY, testX, testY, validX, validY



if __name__ == "__main__":
    data_corpus = "twitter"

    #data preprocessing
    metadata, trainX, trainY, testX, testY, validX, validY = initial_setup(data_corpus)

    # Parameters
    src_len = len(trainX)
    tgt_len = len(trainY)

    # возвращает ошибку если условие ложно
    assert src_len == tgt_len

    batch_size = 32
    n_step = src_len // batch_size
    src_vocab_size = len(metadata['idx2w']) # 8002 (0~8001)
    emb_dim = 1024

    word2idx = metadata['w2idx']   # dict  word 2 index
    idx2word = metadata['idx2w']   # list index 2 word

    unk_id = word2idx['unk']   # 1
    pad_id = word2idx['_']     # 0

    start_id = src_vocab_size  # 8002
    end_id = src_vocab_size + 1  # 8003

    word2idx.update({'start_id': start_id})
    word2idx.update({'end_id': end_id})
    idx2word = idx2word + ['start_id', 'end_id']

    src_vocab_size = tgt_vocab_size = src_vocab_size + 2

    num_epochs = 6
    vocabulary_size = src_vocab_size
    

    # функциия ответа на запрос. 2-ой аргумент принимает количество наиболее вероятных ответов
    def inference(seed, top_n):
        # Установает эту сеть в режиме оценки (если сеть не тренируется)
        model_.eval()
        seed_id = [word2idx.get(w, unk_id) for w in seed.split(" ")]
        sentence_id = model_(inputs=[[seed_id]], seq_length=20, start_token=start_id, top_n = top_n)
        sentence = []
        for w_id in sentence_id[0]:
            w = idx2word[w_id]
            if w == 'end_id':
                break
            sentence = sentence + [w]
        return sentence

    decoder_seq_length = 20
    # создание модели seq2seq
    # encoder и decoder основаны на GRU
    model_ = Seq2seq(
        decoder_seq_length = decoder_seq_length,
        cell_enc=tf.keras.layers.GRUCell,
        cell_dec=tf.keras.layers.GRUCell,
        n_layer=3,
        n_units=256,
        # это таблица соответствия для встраивания слов.
        # Embedding Доступ к содержимому слова осуществляется с помощью целочисленных индексов,
        # а затем вывод представляет собой встроенный вектор слова
        embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim),
        )
    

    # Uncomment below statements if you have already saved the model

    load_weights = tl.files.load_npz(name='model.npz')
    tl.files.assign_weights(load_weights, model_)

    # усовершенствованный градиентный спуск, запомиает напраление предыдущих градиентов
    # которые добавляются к текущему градиенту
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    model_.train()
    # создание списка  фраз на которые отвечает бот
    seeds = ["happy birthday have a nice day",
                 "donald trump won last nights presidential debate according to snap online polls"]

    # model_.load_weights('model.npz')

    for epoch in range(num_epochs):
        model_.train()
        # перемешивает матрицы
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
        total_loss, n_iter = 0, 0
        for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False), 
                        total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs), leave=False):
            # Заполняет последовательности одинаковой длины.
            # Эта функция преобразует список последовательностей num_samples (списки целых чисел)
            # в двумерный массив Numpy формы (num_samples, num_timesteps)
            X = tl.prepro.pad_sequences(X)
            #sequences_add_end_id= Добавление специального маркера конца (id) в конце каждой последовательности.
            _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
            _target_seqs = tl.prepro.pad_sequences(_target_seqs, maxlen=decoder_seq_length)
            _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs, maxlen=decoder_seq_length)
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

            with tf.GradientTape() as tape:
                # compute outputs
                output = model_(inputs = [X, _decode_seqs])
                # Придает массиву новую форму без изменения его данных
                output = tf.reshape(output, [-1, vocabulary_size])

                # compute loss and update model
                # Возвращает выражение кросс-энтропии двух последовательностей
                loss = cross_entropy_seq_with_mask(logits=output, target_seqs=_target_seqs, input_mask=_target_mask)
                # ищем градиент первый аргумент функция, второй значения аргументов функции
                grad = tape.gradient(loss, model_.all_weights)
                # изменение весов модели после подсчёта градиета
                optimizer.apply_gradients(zip(grad, model_.all_weights))
            
            total_loss += loss
            n_iter += 1

        # printing average loss after every epoch
        print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))

        for seed in seeds:
            print("Query >", seed)
            top_n = 3
            for i in range(top_n):
                sentence = inference(seed, top_n)
                print(" >", ' '.join(sentence))

        tl.files.save_npz(model_.all_weights, name='model.npz')


        
    
    
