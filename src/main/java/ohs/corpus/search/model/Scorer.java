package ohs.corpus.search.model;

import ohs.corpus.search.app.DocumentSearcher;
import ohs.corpus.search.index.InvertedIndex;
import ohs.corpus.type.DocumentCollection;
import ohs.matrix.SparseVector;
import ohs.types.generic.Vocab;

public abstract class Scorer {

	// class SearchWorker implements Callable<Map<Integer, SparseVector>> {
	//
	// private List<SparseVector> queryData;
	//
	// private List<SparseVector> scoreData;
	//
	// private DocumentSearcher ds;
	//
	// private AtomicInteger q_cnt;
	//
	// public SearchWorker(DocumentSearcher ds, List<SparseVector> queryData, List<SparseVector> scoreData, AtomicInteger q_cnt) {
	// this.ds = ds;
	// this.queryData = queryData;
	// this.scoreData = scoreData;
	// this.q_cnt = q_cnt;
	// }
	//
	// @Override
	// public Map<Integer, SparseVector> call() throws Exception {
	// int q_loc = 0;
	// Map<Integer, SparseVector> ret = Generics.newHashMap();
	//
	// while ((q_loc = q_cnt.getAndIncrement()) < queryData.size()) {
	// SparseVector Q = queryData.get(q_loc);
	// SparseVector scores = null;
	//
	// if (scoreData == null) {
	// scores = ds.search(Q);
	// } else {
	// SparseVector prevScores = scoreData.get(q_loc);
	// prevScores.sortIndexes();
	// scores = ds.search(Q, prevScores);
	// prevScores.sortValues();
	// }
	// ret.put(q_loc, scores);
	//
	// // if (print_log) {
	// StringBuffer sb = new StringBuffer();
	// sb.append("=====================================");
	// sb.append(String.format("\nThread Name:\t%s", Thread.currentThread().getName()));
	// sb.append("\nNo:\t" + (q_loc + 1));
	// sb.append("\nQ1:\t" + StrUtils.join(" ", ds.getVocab().getObjects(Q.indexes())));
	// sb.append("\nQ2:\t" + VectorUtils.toCounter(Q, ds.getVocab()));
	// sb.append("\nDocs: " + scores.size());
	// System.out.println(sb.toString() + "\n");
	// // }
	//
	// }
	// return ret;
	// }
	//
	// @Override
	// protected void finalize() throws Throwable {
	// super.finalize();
	// rdc.close();
	// }
	// }

	protected Vocab vocab;

	protected DocumentCollection dc;

	protected InvertedIndex ii;

	public Scorer(DocumentSearcher ds) {
		this(ds.getVocab(), ds.getDocumentCollection(), ds.getInvertedIndex());
	}

	public Scorer(Vocab vocab, DocumentCollection dc, InvertedIndex ii) {
		this.vocab = vocab;
		this.dc = dc;
		this.ii = ii;
	}

	public void postprocess(SparseVector scores) {
		scores.sortValues();
	}

	public SparseVector score(SparseVector Q, SparseVector docs, boolean use_index) throws Exception {
		return use_index ? scoreFromIndex(Q, docs) : scoreFromCollection(Q, docs);
	}

	public abstract SparseVector scoreFromCollection(SparseVector Q, SparseVector docs) throws Exception;

	public abstract SparseVector scoreFromIndex(SparseVector Q, SparseVector docs) throws Exception;

	// public List<SparseVector> search(List<SparseVector> qData, List<SparseVector> dData, int thread_size) throws Exception {
	//
	// ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);
	//
	// List<Future<Map<Integer, SparseVector>>> fs = Generics.newArrayList(thread_size);
	//
	// List<Scorer> dss = Generics.newArrayList(thread_size);
	//
	// for (int i = 0; i < thread_size; i++) {
	// dss.add();
	// }
	//
	// AtomicInteger q_cnt = new AtomicInteger(0);
	//
	// for (int i = 0; i < thread_size; i++) {
	// fs.add(tpe.submit(new SearchWorker(dss.get(i), qData, dData, q_cnt)));
	// }
	//
	// List<SparseVector> ret = Generics.newArrayList(qData.size());
	//
	// for (int i = 0; i < qData.size(); i++) {
	// ret.add(new SparseVector(0));
	// }
	//
	// for (int i = 0; i < thread_size; i++) {
	// Map<Integer, SparseVector> res = fs.get(i).get();
	// for (Entry<Integer, SparseVector> e : res.entrySet()) {
	// ret.set(e.getKey(), e.getValue());
	// }
	// }
	// fs.clear();
	//
	// tpe.shutdown();
	//
	// for (int i = 0; i < thread_size; i++) {
	// dss.get(i).close();
	// }
	//
	// return ret;
	// }
}
