https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder

genEpoc = gen_epochs(FLAGS.data_dir, 1, 100, FLAGS.vocab_size, phase="dev")

res = [x for x in genEpoc]

results = [x for x in res[0]]


vocab_file = os.path.join(FLAGS.data_dir,
                              "definitions_%s.vocab" % FLAGS.vocab_size)
embs_dict, pre_emb_dim = load_pretrained_embeddings(FLAGS.embeddings_path)
vocab, _ = data_utils.initialize_vocabulary(vocab_file)
pre_embs = get_embedding_matrix(embs_dict, vocab, pre_emb_dim


(input_node, target_node, predictions, loss, vocab,
          rev_vocab) = restore_model(sess, FLAGS.save_dir, vocab_file,
                                     out_form="cosine")

                                   

gloss, h = results[0]

model_preds = sess.run(predictions, feed_dict={input_node: gloss})


sims = 1 - dist.cdist(model_preds, pre_embs
, metric="cosine")

sims = np.nan_to_num(sims)
top = 50
ranked_ids = sims.argsort(axis = 1)[:, ::-1]
ranks = [np.where(ranked_ids[i] == h[i])[0][0] for i in range(h.shape[0])]


