package ohs.fake;

import java.io.File;
import java.util.Collections;
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
import ohs.io.TextFileWriter;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.feat.select.ChisquareComputer;
import ohs.nlp.ling.types.LDocument;
import ohs.nlp.ling.types.LDocumentCollection;
import ohs.nlp.ling.types.LSentence;
import ohs.nlp.ling.types.LToken;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
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
				File inFile = files.get(file_loc);
				File outFile = new File(inFile.getPath().replace("line", "line_pos"));

				List<String> ins = FileUtils.readLinesFromText(inFile);
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

					LDocument doc = new LDocument();

					for (String str : (title + "\n" + content).split("\n")) {
						if (str.length() == 0) {
							continue;
						}

						LSentence sent = new LSentence();

						for (LNode node : Analyzer.parseJava(str)) {
							Morpheme m = node.morpheme();
							WrappedArray<String> fs = m.feature();
							String[] vals = (String[]) fs.array();
							String word = m.surface();
							String pos = vals[0];

							if (word.length() == 0) {
								continue;
							}

							LToken t = new LToken(2);
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

				FileUtils.writeStringCollectionAsText(outFile.getPath(), outs);
			}
			return (int) 0;
		}

	}

	static {
		Indexer<String> idxer = Generics.newIndexer();
		idxer.add("word");
		idxer.add("pos");

		LToken.INDEXER = idxer;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		DataHandler dh = new DataHandler();
		// dh.tagPos();
		// dh.tagPOS4NaverNews();
		// dh.selectSubset1();
		// dh.extractVocab();

		// dh.formatM2Sentences();
		// dh.makeDicts();
		// dh.makePersonDict();
		// dh.makeOrgaizationDict();
		// dh.makeDicts();
		dh.computeChisquares();

		System.out.println("process ends.");
	}

	public void computeChisquares() throws Exception {
		List<File> files = FileUtils.getFilesUnder(FNPath.NAVER_NEWS_COL_LINE_POS_DIR);

		Collections.reverse(files);

		Vocab vocab = new Vocab();
		Indexer<String> labelIdxer = Generics.newIndexer();

		IntegerArray Y = new IntegerArray(1000000);
		List<SparseVector> X = Generics.newArrayList(100000);

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);
			List<String> lines = FileUtils.readLinesFromText(file);

			for (String line : lines) {
				List<String> ps = StrUtils.split("\t", line);
				ps = StrUtils.unwrap(ps);

				String topic = ps.get(2).replace("/", "-");

				LDocument d = LDocument.newDocument(ps.get(4));

				if (X.size() == 100000) {
					break;
				}

				Y.add(labelIdxer.getIndex(topic));

				Counter<String> c = Generics.newCounter();

				for (LToken t : d.getTokens()) {
					String word = t.getString(0);
					String pos = t.getString(1);

					if (pos.startsWith("N") || pos.startsWith("V") || pos.startsWith("U")) {

					} else {
						continue;
					}

					c.incrementCount(word, 1);
				}

				SparseVector x = VectorUtils.toSparseVector(c, vocab, true);

				X.add(x);
			}

			if (X.size() == 100000) {
				break;
			}
		}

		Y.trimToSize();

		ChisquareComputer cc = new ChisquareComputer(labelIdxer, vocab, new SparseMatrix(X),
				new DenseVector(Y.values()));

		for (int i = 0; i < labelIdxer.size(); i++) {
			String topic = labelIdxer.get(i).replace("/", "-");
			DenseVector c1 = cc.compute(i);
			Counter<String> c2 = VectorUtils.toCounter(c1, vocab);
			c2.keepTopNKeys(5000);

			FileUtils.writeStringCounterAsText(String.format("%s/%s.txt", FNPath.DATA_DIR + "dict_topic", topic), c2);
		}

	}

	public void makeDicts() throws Exception {

		// {
		// List<String> ins =
		// FileUtils.readLinesFromText("../../data/dict/171116_authors.txt");
		// Counter<String> c = Generics.newCounter();
		//
		// for (String s : ins) {
		// String[] ps = s.split("\t");
		// String name = ps[1].trim();
		// if (name.length() == 0) {
		// continue;
		// }
		// c.incrementCount(name, 1);
		// }
		// c.pruneKeysOverThreshold(5);
		//
		// FileUtils.writeStringCollectionAsText(FNPath.DATA_DIR + "pers.txt",
		// Generics.newTreeSet(c.keySet()));
		// }

		{
			List<String> ins = FileUtils.readLinesFromText("../../data/dict/기관 정보 데이터베이스 제공_교육_의료_기타_171116.txt");
			Counter<String> c = Generics.newCounter();

			for (String s : ins) {
				String[] ps = s.split("\t");
				String name = ps[1].trim();
				if (name.length() == 0) {
					continue;
				}
				if (name.startsWith("\"") && name.startsWith("\"")) {
					name = name.substring(1, name.length() - 1);
				}
				c.incrementCount(name, 1);
			}

			FileUtils.writeStringCollectionAsText(FNPath.DATA_DIR + "orgs.txt", Generics.newTreeSet(c.keySet()));
		}
	}

	public void makePersonDict() throws Exception {

		Counter<String> c = Generics.newCounter();

		{
			List<String> ins = FileUtils.readLinesFromText("../../data/dict/171116_authors.txt");

			for (int i = 1; i < ins.size(); i += 2) {
				String s = ins.get(i);
				String[] ps = s.split("\t");
				String kName = ps[1].trim();
				String eName = ps[2].trim();

				if (kName.length() == 0) {
					continue;
				}
				ps = new String[] { kName, eName };
				ps = StrUtils.wrap(ps);
				c.incrementCount(kName, 1);
			}
		}

		{
			List<String> ins = FileUtils.readLinesFromText("../../data/dict/etri_dict/PS_dict.txt", "euc-kr");

			for (int i = 0; i < ins.size(); i++) {
				String s = ins.get(i);
				String[] ps = s.split("\t");
				String kName = ps[0].trim();

				if (kName.length() == 0) {
					continue;
				}

				c.incrementCount(kName, 1);
			}
		}

		FileUtils.writeStringCollectionAsText("../../data/dict/pers.txt", Generics.newTreeSet(c.keySet()));
	}

	public void makeOrgaizationDict() throws Exception {

		Counter<String> c = Generics.newCounter();

		{
			List<String> ins = FileUtils.readLinesFromText("../../data/dict/기관 정보 데이터베이스 제공_교육_의료_기타_171116.txt");

			for (String s : ins) {
				String[] ps = s.split("\t");
				String name = ps[1].trim();
				if (name.length() == 0) {
					continue;
				}
				if (name.startsWith("\"") && name.startsWith("\"")) {
					name = name.substring(1, name.length() - 1);
				}

				if (name.length() > 0) {
					c.incrementCount(name, 1);
				}
			}
		}

		{
			List<String> ins = FileUtils.readLinesFromText("../../data/dict/etri_dict/OG_dict.txt", "euc-kr");

			for (int i = 0; i < ins.size(); i++) {
				String s = ins.get(i);
				String[] ps = s.split("\t");
				String name = ps[0].trim();

				name = StrUtils.normalizeSpaces(name);

				if (name.length() == 0) {
					continue;
				}

				c.incrementCount(name, 1);
			}
		}

		FileUtils.writeStringCollectionAsText("../../data/dict/orgs.txt", Generics.newTreeSet(c.keySet()));
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

				LDocument d = LDocument.newDocument(content);

				Counter<String> c = Generics.newCounter();

				for (LToken t : d.getTokens()) {
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

	public void formatM2Sentences() throws Exception {
		LDocumentCollection c = new LDocumentCollection();

		for (File file : FileUtils.getFilesUnder(FNPath.DATA_DIR + "data")) {
			if (file.getName().startsWith("M2_test_pos")) {
				for (String line : FileUtils.readLinesFromText(file)) {
					List<String> ps = StrUtils.split("\t", line);
					ps = StrUtils.unwrap(ps);

					String label = ps.get(1);
					label = label.equals("0") ? "non-fake" : "fake";
					LDocument d = LDocument.newDocument(ps.get(2));
					d.getAttrMap().put("label", label);
					d.getAttrMap().put("id", ps.get(0));

					c.add(d);
				}
			}
		}

		StringBuffer sb = new StringBuffer();

		for (LDocument d : c) {

			sb.append(String.format("DOCID\t%s", d.getAttrMap().get("id")));
			sb.append(String.format("\nLABEL\t%s", d.getAttrMap().get("label")));
			List<String> l = Generics.newArrayList(d.size());

			for (LSentence s : d) {
				String p1 = StrUtils.join(" ", s.getTokenStrings(0));
				String p2 = "O";
				String[] ps = { p1, p2 };
				ps = StrUtils.wrap(ps);
				l.add(StrUtils.join("\t", ps));
			}

			sb.append("\n" + StrUtils.join("\n", l));
			sb.append("\n\n");
		}

		FileUtils.writeAsText(FNPath.DATA_DIR + "M2_test_sents.txt", sb.toString().trim());

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

			LDocument md = LDocument.newDocument(s);

			CounterMap<String, Integer> cm = Generics.newCounterMap();
			Indexer<String> wordIdxer = Generics.newIndexer();
			List<String> l = Generics.newArrayList(md.size());

			for (int j = 0; j < md.size(); j++) {
				LSentence ms = md.get(j);

				String type = "T";

				if (j != 0) {
					type = "B";
				}

				Counter<Integer> c = cm.getCounter(type);

				StringBuffer sb = new StringBuffer();

				for (LToken t : ms) {
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

	public void selectSubsetNews() throws Exception {
		List<File> files = FileUtils.getFilesUnder(FNPath.NAVER_NEWS_COL_LINE_POS_DIR);

		TextFileWriter writer = new TextFileWriter(FNPath.NAVER_DATA_DIR + "news_2007.txt");

		Counter<String> c = Generics.newCounter();

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);

			if (!file.getName().startsWith("2017")) {
				continue;
			}

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

				c.incrementCount(cat, 1);
			}
			writer.write(StrUtils.join("\n", outs) + "\n");
		}
		writer.close();

		System.out.println(c.toString());
		System.out.println((int) c.totalCount());
	}

	public void tagPos() throws Exception {
		for (File inFile : FileUtils.getFilesUnder(FNPath.DATA_DIR + "data")) {

			// if (!inFile.getName().contains("M2_train")) {
			// continue;
			// }

			if (inFile.getName().contains("pos")) {
				continue;
			}

			File outFile = new File(inFile.getParentFile(), inFile.getName().replace(".txt", "_pos.txt"));
			String input = FileUtils.readFromText(inFile.getPath(), "euc-kr");

			List<String> ins = StrUtils.split("\r\n", input);
			List<String> outs = Generics.newLinkedList();

			for (int i = 1; i < ins.size(); i++) {
				String line = ins.get(i);

				String id = "";
				String title = "";
				String body = "";
				String label = "";
				String corTitle = "";

				List<String> ps = StrUtils.split("\t", line);

				{
					int j = 0;
					id = ps.get(j++);
					title = ps.get(j++);
					body = ps.get(j++);

					if (ps.size() >= 4) {
						label = ps.get(j++);
					}

					if (ps.size() >= 5) {
						corTitle = ps.get(j++);
						corTitle = StrUtils.normalizeSpaces(corTitle);
					}
				}

				{
					StringBuffer sb = new StringBuffer();

					String[] sents = body.replace(".[\\\\s\\u2029]", ".\n").split("\n");

					for (String s : sents) {
						s = StrUtils.normalizeSpaces(s);

						if (s.length() == 0) {
							continue;
						}
						sb.append(s + "\n");
					}
					body = sb.toString().trim();
				}

				LDocumentCollection data = new LDocumentCollection();

				String[] items = new String[] { title + "\n" + body, corTitle };

				for (String item : items) {
					LDocument d = new LDocument();
					for (String p : item.split("\n")) {
						p = p.trim();

						if (p.length() == 0) {
							continue;
						}

						LSentence s = new LSentence();
						for (LNode node : Analyzer.parseJava(p)) {
							Morpheme m = node.morpheme();
							WrappedArray<String> fs = m.feature();
							String[] vals = (String[]) fs.array();
							String word = m.surface();
							String pos = vals[0];

							LToken t = new LToken(2);
							t.add(word);
							t.add(pos);
							s.add(t);
						}
						d.add(s);
					}

					data.add(d);
				}

				List<String> l = Generics.newArrayList(4);

				l.add(id);
				l.add(label);
				l.add(data.get(0).toString().replace("\n", "<nl>"));
				l.add(data.get(1).toString().replace("\n", "<nl>"));

				l = StrUtils.wrap(l);

				outs.add(StrUtils.join("\t", l));
			}

			FileUtils.writeStringCollectionAsText(outFile.getPath(), outs);
		}
	}

	public void tagPosNews() throws Exception {
		// FileUtils.deleteFilesUnder(FNPath.NAVER_NEWS_COL_LINE_POS_DIR);

		AtomicInteger file_cnt = new AtomicInteger(0);

		int thread_size = 4;

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList(thread_size);

		List<File> files = FileUtils.getFilesUnder(FNPath.NAVER_NEWS_COL_LINE_DIR);

		{
			List<File> l = Generics.newArrayList();

			for (File f : files) {
				if (f.getName().startsWith("201707") || f.getName().startsWith("201708")
						|| f.getName().startsWith("201709")) {
					l.add(f);
				}
			}

			files = l;
		}

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new PosTaggingWorker(files, file_cnt)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}

		tpe.shutdown();
	}
}
