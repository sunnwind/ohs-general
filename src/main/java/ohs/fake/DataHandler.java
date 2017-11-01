package ohs.fake;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import org.bitbucket.eunjeon.seunjeon.Analyzer;
import org.bitbucket.eunjeon.seunjeon.LNode;
import org.bitbucket.eunjeon.seunjeon.Morpheme;

import ohs.corpus.type.RawDocumentCollection;
import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.io.TextFileWriter;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.SparseVector;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import scala.collection.mutable.WrappedArray;

public class DataHandler {

	class PosTaggingWorker implements Callable<Integer> {

		private AtomicInteger file_cnt;

		private List<File> files;

		public PosTaggingWorker(List<File> files, AtomicInteger file_cnt) {
			super();
			this.files = files;
			this.file_cnt = file_cnt;
		}

		@Override
		public Integer call() throws Exception {
			int file_loc = 0;
			while ((file_loc = file_cnt.getAndIncrement()) < files.size()) {
				File file = files.get(file_loc);

				List<String> ins = FileUtils.readLinesFromText(file);
				List<String> outs = Generics.newArrayList(ins.size());

				for (int i = 0; i < ins.size(); i++) {
					String line = ins.get(i);

					List<String> ps = StrUtils.split("\t", line);
					ps = StrUtils.unwrap(ps);

					int j = 0;
					String id = ps.get(j++);
					String oid = ps.get(j++);
					String cat = ps.get(j++);
					String date = ps.get(j++);
					String title = ps.get(j++);
					String content = ps.get(j++);
					String url = ps.get(j++);

					content = content.replace(". ", ".\n\n");

					MDocument doc = new MDocument();

					for (String str : (title + "\n" + content).split("\n")) {
						if (str.length() == 0) {
							continue;
						}

						MSentence sent = new MSentence();

						for (LNode node : Analyzer.parseJava(str)) {
							Morpheme m = node.morpheme();
							WrappedArray<String> fs = m.feature();
							String[] vals = (String[]) fs.array();
							String word = m.surface();
							String pos = vals[0];

							if (word.length() == 0) {
								continue;
							}

							MToken t = new MToken(2);
							t.add(word);
							t.add(pos);
							sent.add(t);
						}

						if (sent.size() == 0) {
							continue;
						}

						doc.add(sent);
					}
					doc.trimToSize();

					if (doc.size() == 0) {
						continue;
					}

					// MDocument taggedContent = getText(komoran.analyze(content, 1));

					ps.clear();
					ps.add(id);
					ps.add(oid);
					ps.add(cat);
					ps.add(date);
					ps.add(doc.toString().replace("\n", StrUtils.LINE_REP));
					ps.add(url);

					ps = StrUtils.wrap(ps);

					outs.add(StrUtils.join("\t", ps));
				}

				FileUtils.writeStringCollectionAsText(file.getPath().replace("line", "line_pos"), outs);
			}
			return (int) 0;
		}

	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		DataHandler dh = new DataHandler();
		// dh.tagPOS4NaverNews();
		// dh.selectSubset1();
		// dh.extractVocab();

		dh.test();

		System.out.println("process ends.");
	}

	public void test() throws Exception {
		Counter<String> c = Generics.newCounter();
		TextFileReader reader = new TextFileReader(FNPath.NAVER_DATA_DIR + "news_2007.txt");
		reader.setPrintSize(10000);
		
		while (reader.hasNext()) {
			
			String line = reader.next();
			List<String> ps1 = StrUtils.unwrap(StrUtils.split("\t", line));

			int j = 0;
			String cat = ps1.get(j++);
			String content = ps1.get(j++);
			c.incrementCount(cat, 1);
		}
		reader.printProgress();
		reader.close();

		System.out.println(c.toString());

	}

	public void selectSubset2() throws Exception {
		RawDocumentCollection rdc = new RawDocumentCollection(FNPath.NAVER_NEWS_COL_DC_DIR);

		TextFileWriter writer = new TextFileWriter(FNPath.NAVER_DATA_DIR + "dissam_docs.txt");
		int cnt = 0;

		for (int i = 0; i < rdc.size(); i++) {
			Map<String, String> m = rdc.getMap(i);

			String id = m.get("id");
			String cat = m.get("cat1");
			String title = m.get("title");
			String content = m.get("body");

			String s = title + "\n" + content.replace(StrUtils.LINE_REP, "\n");
			s = s.trim();

			MDocument md = MDocument.newDocument(s);

			CounterMap<String, Integer> cm = Generics.newCounterMap();
			Indexer<String> wordIdxer = Generics.newIndexer();
			List<String> l = Generics.newArrayList(md.size());

			for (int j = 0; j < md.size(); j++) {
				MSentence ms = md.get(j);

				String type = "T";

				if (j != 0) {
					type = "B";
				}

				Counter<Integer> c = cm.getCounter(type);

				StringBuffer sb = new StringBuffer();

				for (MToken t : ms) {
					String word = t.getString(0);
					String pos = t.getString(1);
					sb.append(word + " ");

					if (pos.startsWith("N")) {
						String ss = String.format("%s / %s", t.get(0), t.get(1));
						int w = wordIdxer.getIndex(ss);
						c.incrementCount(w, 1);
					}
				}
				l.add(sb.toString().trim());
			}

			String text = StrUtils.join("\n", l);
			SparseVector sv1 = new SparseVector(cm.getCounter("T"));
			SparseVector sv2 = new SparseVector(cm.getCounter("B"));

			double cosine = VectorMath.cosine(sv1, sv2);

			if (sv1.size() == 0 || sv2.size() == 0) {
				continue;
			}

			if (cosine < 0.1) {

				// System.out.println("=============");
				// System.out.printf("cosine:\t%f\n", cosine);
				// System.out.println(VectorUtils.toCounter(sv1, wordIdxer));
				// System.out.println(VectorUtils.toCounter(sv2, wordIdxer));
				// System.out.println("------------");
				// System.out.println(text);
				// System.out.println();

				StringBuffer sb = new StringBuffer();
				sb.append(String.format("id:\t%s", id));
				sb.append(String.format("\ndseq:\t%d", i));
				sb.append(String.format("\ncat1:\t%s", cat));
				sb.append("\ntext:");
				sb.append("\n" + text);
				sb.append(String.format("\nTV:\t%s", VectorUtils.toCounter(sv1, wordIdxer)));
				sb.append(String.format("\nBV:\t%s", VectorUtils.toCounter(sv2, wordIdxer)));
				sb.append(String.format("\ncosine:\t%f", cosine));

				writer.write(sb + "\n\n");

				if (++cnt == 50000) {
					break;
				}
			}
		}
		rdc.close();
		writer.close();
	}

	static {
		Indexer<String> idxer = Generics.newIndexer();
		idxer.add("word");
		idxer.add("pos");

		MToken.INDEXER = idxer;
	}

	public void extractVocab() throws Exception {
		List<File> files = FileUtils.getFilesUnder(FNPath.NAVER_NEWS_COL_LINE_POS_DIR);
		Counter<String> wordCnts = Generics.newCounter();
		Counter<String> docFreqs = Generics.newCounter();
		int num_docs = 0;

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);

			if (!file.getName().startsWith("2017")) {
				continue;
			}

			// if (num_docs > 10000) {
			// break;
			// }

			List<String> lines = FileUtils.readLinesFromText(file);

			num_docs += lines.size();

			for (String line : lines) {
				List<String> ps = StrUtils.split("\t", line);
				ps = StrUtils.unwrap(ps);

				int j = 0;
				String id = ps.get(j++);
				String oid = ps.get(j++);
				String cat = ps.get(j++);
				String date = ps.get(j++);
				String content = ps.get(j++);
				String url = ps.get(j++);

				MDocument d = MDocument.newDocument(content);

				Counter<String> c = Generics.newCounter();

				for (MToken t : d.getTokens()) {
					c.incrementCount(t.getString(0), 1);
				}

				wordCnts.incrementAll(c);
				docFreqs.incrementAll(c.keySet(), 1);
			}

		}

		Counter<String> tfidfs = Generics.newCounter(wordCnts.size());

		for (String word : wordCnts.keySet()) {
			double cnt = wordCnts.getCount(word);
			double doc_freq = docFreqs.getCount(word);
			double tfidf = TermWeighting.tfidf(cnt, num_docs, doc_freq);
			tfidfs.setCount(word, tfidf);
		}

		TextFileWriter writer = new TextFileWriter(FNPath.NAVER_DATA_DIR + "vocab.txt");

		for (String word : tfidfs.getSortedKeys()) {
			double tfidf = tfidfs.getCount(word);
			int cnt = (int) wordCnts.getCount(word);
			int doc_freq = (int) docFreqs.getCount(word);
			writer.write(String.format("%s\t%d\t%d\t%f\n", word, cnt, doc_freq, tfidf));
		}
		writer.close();

	}

	public void selectSubset1() throws Exception {
		List<File> files = FileUtils.getFilesUnder(FNPath.NAVER_NEWS_COL_LINE_POS_DIR);

		TextFileWriter writer = new TextFileWriter(FNPath.NAVER_DATA_DIR + "news_2007.txt");

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);

			if (!file.getName().startsWith("2017")) {
				continue;
			}

			// if (num_docs > 10000) {
			// break;
			// }

			List<String> ins = FileUtils.readLinesFromText(file);
			List<String> outs = Generics.newLinkedList();

			for (String line : ins) {
				List<String> ps1 = StrUtils.unwrap(StrUtils.split("\t", line));

				int j = 0;
				String id = ps1.get(j++);
				String oid = ps1.get(j++);
				String cat = ps1.get(j++);
				String date = ps1.get(j++);
				String content = ps1.get(j++);
				String url = ps1.get(j++);

				List<String> ps2 = Generics.newArrayList();
				ps2.add(cat);
				ps2.add(content);

				ps2 = StrUtils.wrap(ps2);

				outs.add(StrUtils.join("\t", ps2));
			}
			writer.write(StrUtils.join("\n", outs) + "\n");
		}

		writer.close();
	}

	public void tagPOS4NaverNews() throws Exception {
		FileUtils.deleteFilesUnder(FNPath.NAVER_NEWS_COL_LINE_POS_DIR);

		AtomicInteger file_cnt = new AtomicInteger(0);

		int thread_size = 3;

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList(thread_size);

		List<File> files = FileUtils.getFilesUnder(FNPath.NAVER_NEWS_COL_LINE_DIR);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new PosTaggingWorker(files, file_cnt)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}

		tpe.shutdown();
	}

}
