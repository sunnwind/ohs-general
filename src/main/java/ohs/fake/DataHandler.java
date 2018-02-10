package ohs.fake;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

import org.bitbucket.eunjeon.seunjeon.Analyzer;
import org.bitbucket.eunjeon.seunjeon.LNode;
import org.bitbucket.eunjeon.seunjeon.Morpheme;

import ohs.corpus.type.DocumentCollection;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.io.TextFileWriter;
import ohs.ir.search.app.WordSearcher;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.feat.select.ChisquareComputer;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.nlp.ling.types.LDocument;
import ohs.nlp.ling.types.LDocumentCollection;
import ohs.nlp.ling.types.LSentence;
import ohs.nlp.ling.types.LToken;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.types.generic.SetMap;
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
		// dh.selectNewsSubset();
		// dh.buildM1ExtraData();
		// dh.buildM2ExtraData();

		dh.tagPos();
		// dh.tagPOS4NaverNews();
		// dh.selectSubset1();
		// dh.selectSubsetNews2();
		// dh.extractVocab();

		// dh.formatM2Sentences();
		// dh.makeDicts();
		// dh.makePersonDict();
		// dh.makeOrgaizationDict();
		// dh.computeChisquares();
		// dh.makeTopicDicts();
		// dh.getM2Labels();
		// dh.test();

		System.out.println("process ends.");
	}
	


	public void test() throws Exception {
		String text = FileUtils.readFromText(FNPath.DATA_DIR + "M1_tagging/연예.txt");
		String text2 = text.replace("Label:", "\nLabel:").trim();

		// text2 = text2.replace("\n", "<nl>");
		// text2 = text2.replace("\t", "<tab>");
		//
		// text2 = StrUtils.normalizeSpaces(text2);
		// text2 = text2.replace("<nl>", "\n");
		// text2 = text2.replace("<tab>", "\t");

		FileUtils.writeAsText(FNPath.DATA_DIR + "M1_tagging/연예-2.txt", text2.trim());
	}

	public void computeChisquares() throws Exception {
		List<File> files = FileUtils.getFilesUnder(FNPath.NAVER_NEWS_COL_LINE_POS_DIR);

		Collections.reverse(files);

		Vocab vocab = new Vocab();
		Indexer<String> labelIdxer = Generics.newIndexer();

		IntegerArray Y = new IntegerArray(1000000);
		List<SparseVector> X = Generics.newArrayList(100000);

		int doc_size = 1000000;

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);
			List<String> lines = FileUtils.readLinesFromText(file);

			for (String line : lines) {
				List<String> ps = StrUtils.split("\t", line);
				ps = StrUtils.unwrap(ps);

				String topic = ps.get(2).replace("/", "-");

				LDocument d = LDocument.newDocument(ps.get(4));

				if (X.size() == doc_size) {
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

			if (X.size() == doc_size) {
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

	public void makeTopicDicts() throws Exception {
		CounterMap<String, String> cm = Generics.newCounterMap();

		List<File> files = FileUtils.getFilesUnder(FNPath.DATA_DIR + "dict_topic");

		for (File file : files) {
			String fileName = file.getName();
			String topic = fileName.split("\\.")[0];
			Counter<String> c = FileUtils.readStringCounterFromText(file.getPath());
			cm.setCounter(topic, c);
		}

		cm = cm.invert();

		List<String> words = Generics.newArrayList(cm.keySet());

		for (int i = 0; i < words.size(); i++) {
			String word = words.get(i);

			if (cm.getCounter(word).size() == files.size()) {
				cm.removeKey(word);
			}
		}

		cm = cm.invert();

		for (String topic : cm.keySet()) {
			String outFileName = FNPath.DATA_DIR + "dict_topic2/" + topic + ".txt";
			FileUtils.writeStringCounterAsText(outFileName, cm.getCounter(topic));
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

	public void getM2Labels() throws Exception {

		List<String> lines = FileUtils.readLinesFromText(FNPath.DATA_DIR + "M2_test_sents_labeled.txt");

		List<String> outs = Generics.newArrayList(lines.size());

		for (int i = 0; i < lines.size();) {
			if (lines.get(i).startsWith("DOCID")) {
				String docid = lines.get(i).split("\t")[1];

				int j = 0;
				for (int k = i + 1; k < lines.size(); k++) {
					if (lines.get(k).startsWith("DOCID")) {
						j = k;
						break;
					}
				}

				if (j == 0) {
					j = lines.size() - 1;
				}

				int label = 0;

				String s = StrUtils.join("\n", lines, i, j);

				if (s.contains("\tX") || s.contains("\tx")) {
					label = 1;
				}

				outs.add(String.format("%s\t%d", docid, label));

				i = j;
			} else {
				i++;
			}
		}

		System.out.println(StrUtils.join("\n", outs));

		StringBuffer sb = new StringBuffer();

		// for (LDocument d : c) {
		//
		// sb.append(String.format("DOCID\t%s", d.getAttrMap().get("id")));
		// sb.append(String.format("\nLABEL\t%s", d.getAttrMap().get("label")));
		// List<String> l = Generics.newArrayList(d.size());
		//
		// for (LSentence s : d) {
		// String p1 = StrUtils.join(" ", s.getTokenStrings(0));
		// String p2 = "O";
		// String[] ps = { p1, p2 };
		// ps = StrUtils.wrap(ps);
		// l.add(StrUtils.join("\t", ps));
		// }
		//
		// sb.append("\n" + StrUtils.join("\n", l));
		// sb.append("\n\n");
		// }

		// FileUtils.writeAsText(FNPath.DATA_DIR + "M2_test_sents.txt",
		// sb.toString().trim());
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

	public void buildM1ExtraData() throws Exception {

		List<String> res = Generics.newArrayList();

		for (File file : FileUtils.getFilesUnder(FNPath.DATA_DIR + "M1_tagging")) {
			String s = FileUtils.readFromText(file);
			s = s.replace("\r", "");
			List<String> lines = StrUtils.split("\n\n", s);

			for (int i = 0; i < lines.size(); i++) {
				String line = lines.get(i);

				List<String> lines2 = StrUtils.split("\n", line);

				String id = "";
				String label = "";
				String topic = "";

				String title = "";
				String fakeTitle = "";
				List<String> sents = Generics.newArrayList();

				boolean is_valid = true;

				for (int j = 0; j < lines2.size(); j++) {
					List<String> ps = StrUtils.split("\t", lines2.get(j));

					if (ps.size() != 2) {
						is_valid = false;
						break;
					}

					String val = ps.get(1);

					if (j == 0) {
						label = val;
					} else if (j == 1) {
						topic = val;
					} else if (j == 2) {
						id = val;
					} else if (j == 3) {
						title = val;
					} else if (j == 4) {
						fakeTitle = val;
					} else {
						sents.add(val);
					}
				}

				if (!is_valid) {
					continue;
				}

				Counter<String> c = Generics.newCounter();

				for (String sent : sents) {
					for (String t : StrUtils.split(" ", sent)) {
						c.incrementCount(t, 1);
					}
				}

				if (c.totalCount() > 10000) {
					System.out.println();
				}

				if (label.equals("0")) {
					title = "";
				}

				List<String> vals = Generics.newArrayList();
				vals.add(id);
				vals.add(fakeTitle);
				vals.add(StrUtils.join("<nl>", sents));
				vals.add(label);
				vals.add(title);
				res.add(StrUtils.join("\t", vals));
			}
		}

		String out = "식별자\t뉴스제목\t뉴스본문\tLabel\t정답뉴스제목" + "\n" + StrUtils.join("\n", res);

		FileUtils.writeAsText(FNPath.DATA_DIR + "data/M1_train-3.txt", out);
	}

	public void buildM2ExtraData() throws Exception {

		List<String> res = Generics.newArrayList();

		for (File file : FileUtils.getFilesUnder(FNPath.DATA_DIR + "M2_tagging")) {
			String s = FileUtils.readFromText(file);
			List<String> lines = StrUtils.split("\n\n", s);

			for (int i = 0; i < lines.size(); i++) {
				String line = lines.get(i);

				List<String> lines2 = StrUtils.split("\n", line);

				String id = "";
				String label = "";
				String topic = "";

				String title = "";
				List<String> sents = Generics.newArrayList();

				for (int j = 0; j < lines2.size(); j++) {
					List<String> ps = StrUtils.split("\t", lines2.get(j));
					String val = ps.get(1);

					if (j == 0) {
						label = val;
					} else if (j == 1) {
						topic = val;
					} else if (j == 2) {
						id = val;
					} else if (j == 3) {
						title = val;
					} else if (j == 4) {
					} else {
						sents.add(val);
					}
				}

				List<String> vals = Generics.newArrayList();
				vals.add(id);
				vals.add(title);
				vals.add(StrUtils.join("<nl>", sents));
				vals.add(label);
				res.add(StrUtils.join("\t", vals));
			}
		}

		String out = "식별자\t뉴스제목\t뉴스본문\tLabel" + "\n" + StrUtils.join("\n", res);

		FileUtils.writeAsText(FNPath.DATA_DIR + "data/M2_train-3.txt", out);
	}

	public void selectNewsSubset() throws Exception {

		List<File> files = FileUtils.getFilesUnder(FNPath.NAVER_NEWS_COL_LINE_DIR);

		Collections.reverse(files);

		SetMap<String, String> sm = Generics.newSetMap();

		int col_size = 1000000;

		String regex = "[\\p{Punct}\\s\u2029]+";

		int doc_cnt = 0;
		int num_files = 10;

		for (int i = 0; i < files.size() && i < num_files; i++) {
			File file = files.get(i);

			for (String line : FileUtils.readLinesFromText(file)) {
				List<String> ps = StrUtils.split("\t", line);
				ps = StrUtils.unwrap(ps);

				String id = ps.get(0);
				String topic = ps.get(2);
				String title = ps.get(4);
				String body = ps.get(5);

				String s = title + "\n" + body;

				Counter<String> c = Generics.newCounter();

				for (String t : s.split(regex)) {
					c.incrementCount(t, 1);
				}

				if (c.totalCount() < 100) {
					continue;
				}

				sm.put(topic, id);

				if (doc_cnt++ == col_size) {
					break;
				}
			}

			if (doc_cnt == col_size) {
				break;
			}
		}

		for (String topic : sm.keySet()) {
			Set<String> set = sm.get(topic);

			List<String> list = Generics.newArrayList(set);
			Collections.shuffle(list);

			list = list.subList(0, Math.min(list.size(), 1000));
			sm.put(topic, Generics.newHashSet(list));
		}

		ListMap<String, String> lm = Generics.newListMap();

		for (int i = 0; i < files.size() && i < num_files; i++) {
			File file = files.get(i);

			for (String line : FileUtils.readLinesFromText(file)) {
				List<String> ps = StrUtils.split("\t", line);
				ps = StrUtils.unwrap(ps);

				String id = ps.get(0);
				String topic = ps.get(2);
				String title = ps.get(4);
				String body = ps.get(5);

				if (sm.contains(topic, id)) {
					body = body.replace(".\" ", ".\"\n");
					body = body.replace(". ", ".\n");
					String s = title + "\n" + body;
					lm.put(topic, id + "\t" + s);
				}
			}
		}

		for (String topic : lm.keySet()) {
			List<String> ins = lm.get(topic);
			List<String> outs = Generics.newArrayList();

			for (int i = 0; i < ins.size(); i++) {
				List<String> ps = StrUtils.split("\t", ins.get(i));
				String id = ps.get(0);
				String s = ps.get(1);

				topic = topic.replace("/", "-");

				StringBuffer sb = new StringBuffer();
				sb.append(String.format("Label:\t%d", 0));
				sb.append(String.format("\nTopic:\t%s", topic));
				sb.append(String.format("\nID:\t%s", id));

				List<String> sents = StrUtils.split("\n", s);

				for (int j = 0; j < sents.size(); j++) {
					if (j == 0) {
						sb.append(String.format("\nT:\t%s", sents.get(j)));
						sb.append(String.format("\nFT:\t%s", sents.get(j)));
					} else {
						sb.append(String.format("\nB:\t%s", sents.get(j)));
					}

				}

				outs.add(sb.toString());
			}

			FileUtils.writeAsText(String.format("%s/news_subset/%s.txt", FNPath.DATA_DIR, topic),
					StrUtils.join("\n\n", outs));

		}
	}

	public void tagPos() throws Exception {
		for (File inFile : FileUtils.getFilesUnder(FNPath.DATA_DIR + "data")) {
			int data_type = 0;

			if (!inFile.getName().contains("test")) {
				continue;
			}

			String encoding = "euc-kr";
			String delim = "\r\n";

			if (data_type == 1) {
				encoding = "UTF-8";
				delim = "\n";
			}

			File outFile = new File(FNPath.DATA_DIR + "data_pos", inFile.getName().replace(".txt", "_pos.txt"));
			String input = FileUtils.readFromText(inFile.getPath(), "UTF-8");

			List<String> ins = StrUtils.split(delim, input);
			List<String> outs = Generics.newLinkedList();

			if (data_type == 1) {
				for (int i = 1; i < ins.size(); i++) {
					String in = ins.get(i);
					String text = "";

					String id = "";
					String title = "";
					String body = "";
					String label = "";
					String corTitle = "";

					List<String> ps = StrUtils.split(",", in);

					// System.out.println(ps.toString());

					{
						int k = 0;
						id = ps.get(k++);
						title = ps.get(k++);
						body = ps.get(k++);

						StringBuffer sb = new StringBuffer();

						for (String p : body.split("<nl>")) {
							p = p.trim();
							if (p.length() == 0) {
								continue;
							}
							sb.append(p + "\n");
						}

						body = sb.toString().trim();
						// System.out.printf("[%s]\n", body);

						if (ps.size() >= 4) {
							label = ps.get(k++);
						}

						if (ps.size() >= 5) {
							corTitle = ps.get(k++);
							corTitle = StrUtils.normalizeSpaces(corTitle);
						}
					}

					{
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
									Morpheme mm = node.morpheme();
									WrappedArray<String> fs = mm.feature();
									String[] vals = (String[]) fs.array();
									String word = mm.surface();
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
				}
				FileUtils.writeStringCollectionAsText(outFile.getPath(), outs);
			} else if (data_type == 0) {
				Pattern m = Pattern.compile("^\\d_\\d+");
				for (int i = 1; i < ins.size();) {
					String in1 = ins.get(i);
					String text = "";

					if (m.matcher(ins.get(i)).find()) {
						int j = 0;

						for (int k = i + 1; k < ins.size(); k++) {
							String in2 = ins.get(k);
							if (m.matcher(ins.get(k)).find()) {
								j = k;
								break;
							}
						}

						if (j == 0) {
							j = i + 1;
						}

						text = StrUtils.join("\n", ins, i, j);
						text = text.replace(". ", ".\n");
						text = text.replace("\n", "<nl>");
						i = j;

						String id = "";
						String title = "";
						String body = "";
						String label = "";
						String corTitle = "";

						List<String> ps = StrUtils.split("\t", text);

						System.out.println(ps.toString());

						{
							int k = 0;
							id = ps.get(k++);
							title = ps.get(k++);
							body = ps.get(k++);

							StringBuffer sb = new StringBuffer();

							for (String p : body.split("<nl>")) {
								p = p.trim();
								if (p.length() == 0) {
									continue;
								}
								sb.append(p + "\n");
							}

							body = sb.toString().trim();
							// System.out.printf("[%s]\n", body);

							if (ps.size() >= 4) {
								label = ps.get(k++);
							}

							if (ps.size() >= 5) {
								corTitle = ps.get(k++);
								corTitle = StrUtils.normalizeSpaces(corTitle);
							}
						}

						{
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
										Morpheme mm = node.morpheme();
										WrappedArray<String> fs = mm.feature();
										String[] vals = (String[]) fs.array();
										String word = mm.surface();
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

					} else {
						i++;
					}
				}
				FileUtils.writeStringCollectionAsText(outFile.getPath(), outs);
			}
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
