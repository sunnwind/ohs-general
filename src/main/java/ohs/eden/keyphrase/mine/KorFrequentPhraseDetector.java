package ohs.eden.keyphrase.mine;

import java.io.File;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

import ohs.corpus.type.DataCompression;
import ohs.corpus.type.DocumentCollection;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.FileUtils;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.ListList;
import ohs.types.generic.ListMap;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.types.number.LongArray;
import ohs.utils.Generics;
import ohs.utils.Generics.ListType;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class KorFrequentPhraseDetector {

	public class DetectWorker implements Callable<Integer> {

		private DocumentCollection dc;

		private List<File> files;

		private AtomicInteger file_cnt;

		private Timer timer;

		public DetectWorker(DocumentCollection dc, List<File> files, AtomicInteger file_cnt, Timer timer) {
			this.dc = dc;
			this.files = files;
			this.file_cnt = file_cnt;
			this.timer = timer;
		}

		@Override
		public Integer call() throws Exception {
			int file_loc = 0;

			while ((file_loc = file_cnt.getAndIncrement()) < files.size()) {
				File inFile = files.get(file_loc);

				// if (file_loc >= 1) {
				// break;
				// }

				Map<String, LongArray> index = createIndex(inFile);

				File outFile = new File(tmpDir2, inFile.getName());
				FileChannel out = FileUtils.openFileChannel(outFile, "rw");
				int phrs_len = 1;

				while (index.size() != 0 && phrs_len++ != max_phrs_len) {
					ListMap<Integer, Integer> docToLocs = Generics.newListMap(ListType.LINKED_LIST);

					for (String phrs : index.keySet()) {
						LongArray poss = index.get(phrs);

						for (int k = 0; k < poss.size(); k++) {
							long pos = poss.get(k);
							long[] two = getTwoIndexes(pos);
							int dseq = (int) two[0];
							int wloc = (int) two[1];
							docToLocs.put(dseq, wloc);
						}
					}
					index.clear();

					for (int dseq : docToLocs.keySet()) {
						List<Integer> wlocs = docToLocs.get(dseq);
						docToLocs.put(dseq, Generics.newArrayList(wlocs));
					}

					IntegerArray dseqs = new IntegerArray(docToLocs.keySet());
					dseqs.sort(false);

					ListMap<String, Long> tmpIndex = Generics.newListMap(ListType.LINKED_LIST);

					for (int j = 0; j < dseqs.size(); j++) {
						int dseq = dseqs.get(j);
						IntegerArray d = dc.get(dseq).getSecond();

						if (d.size() > max_doc_len) {
							d = d.subArray(0, max_doc_len);
						}

						IntegerArray starts = new IntegerArray(docToLocs.get(dseq));
						starts.sort(false);

						for (int start : starts) {
							int end = start + phrs_len;

							if (end >= d.size()) {
								continue;
							}

							IntegerArray sub = d.subArray(start, end);
							int sent_end_cnt = 0;

							for (int w : sub) {
								if (w == DocumentCollection.SENT_END) {
									sent_end_cnt++;
								}
							}

							if (sent_end_cnt > 0) {
								continue;
							}

							String phrs = StrUtils.join("_", vocab.getObjects(sub));
							long pos = getSingleIndex(dseq, start);
							tmpIndex.put(phrs, pos);
						}
					}
					docToLocs.clear();

					for (String phrs : tmpIndex.keySet()) {
						List<Long> poss = tmpIndex.get(phrs);
						if (poss.size() < min_support) {
							continue;
						}
						index.put(phrs, new LongArray(poss));
					}
					tmpIndex.clear(false);

					write(index, out);
				}
				out.close();

				int prog = BatchUtils.progress(file_loc + 1, files.size());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, file_loc + 1, files.size(), timer.stop());
				}
			}
			return null;
		}

		private Map<String, LongArray> createIndex(File file) throws Exception {
			FileChannel fc = FileUtils.openFileChannel(file, "rw");
			ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc);
			fc.close();

			int i = 0;
			IntegerArray ws = ByteArrayUtils.toIntegerArray(data.get(i++));
			Map<String, LongArray> ret = Generics.newHashMap(ws.size());

			for (int j = 0; j < ws.size(); j++) {
				int w = ws.get(j);
				String word = vocab.getObject(w);
				LongArray poss = ByteArrayUtils.toLongArray(data.get(i++));
				ret.put(word, poss);
			}
			return ret;
		}

		@Override
		protected void finalize() throws Throwable {
			super.finalize();
			dc.close();
		}

		private void write(Map<String, LongArray> index, FileChannel out) throws Exception {
			List<String> phrss = Generics.newArrayList(index.keySet());
			Collections.sort(phrss);

			for (int i = 0; i < phrss.size(); i++) {
				String phrs = phrss.get(i);

				List<String> words = StrUtils.split("_", phrs);
				IntegerArray ws = new IntegerArray(words.size());

				for (String word : words) {
					int w = vocab.indexOf(word);
					if (w >= 0) {
						ws.add(w);
					}
				}

				if (words.size() != ws.size()) {
					continue;
				}

				int w_at_start = ws.get(0);
				int w_at_end = ws.get(ws.size() - 1);

				String taggedWord_at_start = vocab.getObject(w_at_start);
				String taggedWord_at_end = vocab.getObject(w_at_end);

				String[] parts = StrUtils.split2Two("/", taggedWord_at_end);

				if (parts.length != 2) {
					continue;
				}
				String word_at_end = parts[0];
				String tag_at_end = parts[1];

				if (stopwords.contains(w_at_end) || stoptags.contains(tag_at_end) || puncPat.matcher(word_at_end).find()) {
					continue;
				}

				ListMap<Integer, Integer> docToLocs = Generics.newListMap(ListType.LINKED_LIST);

				LongArray poss = index.get(phrs);
				for (long pos : poss) {
					long[] two = getTwoIndexes(pos);
					int dseq = (int) two[0];
					int wloc = (int) two[1];
					docToLocs.put(dseq, wloc);
				}

				IntegerArray dseqs = new IntegerArray(docToLocs.keySet());
				dseqs.sort(false);

				IntegerArrayMatrix locData = new IntegerArrayMatrix(dseqs.size());
				for (int dseq : dseqs) {
					IntegerArray wlocs = new IntegerArray(docToLocs.get(dseq));
					wlocs.sort(false);

					locData.add(wlocs);
				}
				docToLocs.clear();

				SPostingList pl = new SPostingList(phrs, dseqs, locData);
				SPostingList.writePostingList(pl, out, false);
			}
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		{
			Set<String> stoptags = Generics.newHashSet();
			stoptags.add("SS");
			stoptags.add("SP");
			stoptags.add("SE");
			stoptags.add("SF");

			stoptags.add("ETM");
			stoptags.add("ETN");
			stoptags.add("EC");
			stoptags.add("EP");
			stoptags.add("EF");

			stoptags.add("XSN");
			stoptags.add("XSV");
			stoptags.add("XSA");
			stoptags.add("JKO");
			stoptags.add("JKB");
			stoptags.add("JX");
			stoptags.add("VX");
			stoptags.add("VCP");
			stoptags.add("VCP");
			stoptags.add("VCN");

			String dir = "../../data/naver_news/col/dc/";
			KorFrequentPhraseDetector fpd = new KorFrequentPhraseDetector(dir, 6, 30, null, stoptags);
			fpd.setOutputFileSize(500);
			fpd.setThreadSize(2);
			fpd.detect(false);
		}

		// {
		//
		// Set<String> stopwords = FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);
		//
		// String[] dirs = { MIRPath.OHSUMED_COL_DC_DIR, MIRPath.TREC_CDS_2016_COL_DC_DIR, MIRPath.BIOASQ_COL_DC_DIR,
		// MIRPath.WIKI_COL_DIR };
		//
		// for (int i = 0; i < dirs.length; i++) {
		// if (i != 1) {
		// continue;
		// }
		//
		// String dir = dirs[i];
		// FrequentPhraseDetector fpd = new FrequentPhraseDetector(dir, 6, 30, stopwords);
		// fpd.setOutputFileSize(500);
		// fpd.setThreadSize(1);
		// fpd.detect(true);
		// }
		// }

		System.out.println("process ends.");
	}

	private DocumentCollection dc;

	private Vocab vocab;

	private Set<Integer> stopwords;

	private Pattern puncPat = Pattern.compile("^\\p{Punct}+");

	private int max_doc_len = 100000;

	private int max_buf_size = FileUtils.DEFAULT_BUF_SIZE * 2;

	private int output_file_size = 100;

	private int max_phrs_len = 6;

	private int min_support = 5;

	private int thread_size = 1;

	private File tmpDir1;

	private File tmpDir2;

	private File dataDir;

	private File phrsDir;

	private Set<String> stoptags;

	public KorFrequentPhraseDetector(String dataDir, int max_phrs_len, int min_support, Set<String> stopwords, Set<String> stoptags)
			throws Exception {
		this.dataDir = new File(dataDir);
		this.dc = new DocumentCollection(dataDir);
		this.max_phrs_len = max_phrs_len;
		this.min_support = min_support;
		this.stopwords = Generics.newHashSet();
		this.stoptags = Generics.newHashSet();

		if (stopwords != null && stopwords.size() > 0) {
			for (String word : stopwords) {
				int w = dc.getVocab().indexOf(word);
				if (w >= 0) {
					this.stopwords.add(w);
				}
			}
		}

		if (stoptags != null) {
			for (String tag : stoptags) {
				this.stoptags.add(tag.toLowerCase());
			}
		}

		phrsDir = new File(dataDir, "phrs");

		tmpDir1 = new File(phrsDir, "tmp-index/");

		tmpDir2 = new File(phrsDir, "tmp-phrs/");

		vocab = dc.getVocab();
	}

	private void createIndex() throws Exception {
		System.out.println("create index.");
		Timer timer = Timer.newTimer();

		ListList<Integer> wsData = Generics.newListList(10, ListType.ARRAY_LIST, ListType.LINKED_LIST);
		ListList<Long> posData = Generics.newListList(10, ListType.ARRAY_LIST, ListType.LINKED_LIST);

		List<FileChannel> outs = Generics.newArrayList(output_file_size);

		for (int i = 0; i < output_file_size; i++) {
			DecimalFormat df = new DecimalFormat("000000");
			FileChannel fc = FileUtils.openFileChannel(new File(tmpDir1, df.format(i) + ".ser"), "rw");
			outs.add(fc);
		}

		int[][] ranges = BatchUtils.getBatchRanges(dc.size(), 1000);
		int doc_cnt = 0;
		int buf_size = 0;

		for (int i = 0; i < ranges.length; i++) {
			int[] range = ranges[i];

			List<Pair<String, IntegerArray>> ps = dc.getRange(range);

			for (int dseq = range[0], k = 0; dseq < range[1]; dseq++, k++) {
				IntegerArray d = ps.get(k).getSecond();

				for (int wloc = 0; wloc < d.size() && wloc < max_doc_len; wloc++) {
					int w = d.get(wloc);
					long pos = getSingleIndex(dseq, wloc);

					if (pos < 0 || pos >= Long.MAX_VALUE) {
						System.out.printf("[%d] is not valid at [%d, %d]\n", pos, dseq, wloc);
						System.exit(0);
					}

					if (w == DocumentCollection.SENT_END || stopwords.contains(w) || vocab.getCount(w) < min_support) {
						continue;
					}

					String s = vocab.getObject(w);
					String[] parts = StrUtils.split2Two("/", s);

					if (parts.length != 2) {
						continue;
					}

					String word = parts[0];
					String tag = parts[1];

					if (puncPat.matcher(word).find()) {
						continue;
					}

					if (stoptags.contains(tag)) {
						continue;
					}

					int file_loc = w % output_file_size;

					wsData.get(file_loc, true).add(w);
					posData.get(file_loc, true).add(pos);

					buf_size += Integer.BYTES + Long.BYTES;

					if (buf_size >= max_buf_size) {
						write(wsData, posData, outs);
						buf_size = 0;
					}
				}

				int prog = BatchUtils.progress(++doc_cnt, dc.size());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, doc_cnt, dc.size(), timer.stop());
				}
			}
		}

		if (wsData.size() > 0) {
			write(wsData, posData, outs);
			buf_size = 0;
		}

	}

	public void detect(boolean create_index) throws Exception {
		System.out.println("detect.");
		if (create_index) {
			FileUtils.deleteFilesUnder(tmpDir1);
			createIndex();
			mergeIndex();
		}
		detectPhrases();
		mergePhrases();
	}

	private void detectPhrases() throws Exception {
		System.out.println("detect phrases.");

		FileUtils.deleteFilesUnder(tmpDir2);

		List<File> files = FileUtils.getFilesUnder(tmpDir1);

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList(thread_size);

		AtomicInteger file_cnt = new AtomicInteger(0);

		Timer timer = Timer.newTimer();

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new DetectWorker(dc.copyShallow(), files, file_cnt, timer)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}
		fs.clear();

		tpe.shutdown();
	}

	private long getSingleIndex(long dseq, long wloc) {
		return dseq * max_doc_len + wloc;
	}

	private long[] getTwoIndexes(long i) {
		long dseq = (i / max_doc_len);
		long wloc = (i % max_doc_len);
		return new long[] { dseq, wloc };
	}

	private void mergeIndex() throws Exception {
		System.out.println("merge index.");
		Timer timer = Timer.newTimer();

		List<File> files = FileUtils.getFilesUnder(tmpDir1);

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);
			ListMap<Integer, Long> lm = Generics.newListMap(ListType.LINKED_LIST);

			{
				FileChannel fc = FileUtils.openFileChannel(file, "r");

				while (fc.position() < fc.size()) {
					ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc);
					IntegerArray ws = ByteArrayUtils.toIntegerArray(data.get(0));
					LongArray poss = ByteArrayUtils.toLongArray(data.get(1));

					for (int j = 0; j < ws.size(); j++) {
						int w = ws.get(j);
						long pos = poss.get(j);
						lm.put(w, pos);
					}
				}
				fc.close();
			}

			{
				IntegerArray ws = new IntegerArray(lm.keySet());
				ws.sort(false);

				ByteArrayMatrix data = new ByteArrayMatrix(ws.size() + 1);
				data.add(ByteArrayUtils.toByteArray(ws));

				for (int w : ws) {
					LongArray poss = new LongArray(lm.get(w));
					data.add(ByteArrayUtils.toByteArray(poss));
				}

				file.delete();
				FileChannel fc = FileUtils.openFileChannel(file, "rw");
				FileUtils.write(data, fc);
				fc.close();

				int prog = BatchUtils.progress(i + 1, files.size());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, i + 1, files.size(), timer.stop());
				}
			}
			lm.clear();
		}
	}

	private void mergePhrases() throws Exception {
		System.out.println("merge phrases.");

		List<Long> starts = Generics.newLinkedList();
		List<Integer> lens = Generics.newLinkedList();
		List<Integer> phrsLens = Generics.newLinkedList();
		Counter<Integer> c = Generics.newCounter();

		{
			Timer timer = Timer.newTimer();

			File outFile = new File(dataDir, PhraseCollection.NAME);
			outFile.delete();

			FileChannel out = FileUtils.openFileChannel(outFile, "rw");
			List<File> files = FileUtils.getFilesUnder(tmpDir2);

			for (int i = 0; i < files.size(); i++) {
				File file = files.get(i);
				FileChannel in = FileUtils.openFileChannel(file, "r");

				while (in.position() < in.size()) {
					SPostingList pl = SPostingList.readPostingList(in, false);
					long[] info = SPostingList.writePostingList(pl, out, true);
					starts.add(info[0]);
					lens.add((int) info[1]);
					phrsLens.add(pl.getPhrase().split("_").length);
					c.incrementCount(pl.getPhrase().split("_").length, 1);
				}
				in.close();

				int prog = BatchUtils.progress(i + 1, files.size());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, i + 1, files.size(), timer.stop());
				}
			}
			out.close();

			System.out.println(c.toString());
		}

		{
			LongArray s = new LongArray(starts);
			IntegerArray l1 = new IntegerArray(lens);
			IntegerArray l2 = new IntegerArray(phrsLens);

			DataCompression.encodeGaps(s);
			ByteArrayMatrix data = new ByteArrayMatrix(3);
			data.add(DataCompression.encode(s));
			data.add(DataCompression.encode(l1));
			data.add(DataCompression.encode(l2));

			File outFile = new File(dataDir, PhraseCollection.META_NAME);
			FileChannel out = FileUtils.openFileChannel(outFile, "rw");
			FileUtils.write(data, out);
			out.close();
		}

		FileUtils.deleteFilesUnder(tmpDir1);
		FileUtils.deleteFilesUnder(tmpDir2);

	}

	public void setMaxDocLen(int max_doc_len) {
		this.max_doc_len = max_doc_len;
	}

	public void setMaxPhraseLength(int max_phrs_len) {
		this.max_phrs_len = max_phrs_len;
	}

	public void setMinSupport(int min_supoort) {
		this.min_support = min_supoort;
	}

	public void setOutputFileSize(int output_file_size) {
		this.output_file_size = output_file_size;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

	private void write(ListList<Integer> wsData, ListList<Long> posData, List<FileChannel> outs) throws Exception {
		for (int i = 0; i < output_file_size; i++) {
			List<Integer> ws = wsData.get(i);
			List<Long> poss = posData.get(i);
			FileChannel out = outs.get(i);

			if (ws.size() > 0 && poss.size() > 0) {
				ByteArrayMatrix data = new ByteArrayMatrix(2);
				data.add(ByteArrayUtils.toByteArray(new IntegerArray(ws)));
				data.add(ByteArrayUtils.toByteArray(new LongArray(poss)));
				FileUtils.write(data, out);
			}
		}
		wsData.clear();
		posData.clear();
	}

}
