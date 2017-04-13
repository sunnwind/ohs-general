package ohs.eden.keyphrase.mine;

import java.io.File;
import java.lang.Character.UnicodeBlock;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;
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
import ohs.ir.medical.general.MIRPath;
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
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class FrequentPhraseDetector {

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

		public Integer call() throws Exception {
			callNew();
			return null;
		}

		public Integer callNew() throws Exception {
			int file_loc = 0;

			while ((file_loc = file_cnt.getAndIncrement()) < files.size()) {
				File inFile = files.get(file_loc);

				ListMap<String, Long> index = createLocalIndex(inFile);

				File outFile = new File(tmpDir2, inFile.getName());
				FileChannel out = FileUtils.openFileChannel(outFile, "rw");
				int phrs_len = 1;

				while (index.size() != 0 && phrs_len++ != max_phrs_len) {
					ListMap<Integer, Integer> docToLocs = getDocToLocs(index);

					int index_size = index.size();

					index.clear();

					ListMap<String, Long> newIndex = getNewIndex(docToLocs, phrs_len, index_size);

					docToLocs.clear();

					for (String phrs : newIndex.keySet()) {
						List<Long> poss = newIndex.get(phrs);
						if (poss.size() < min_support) {
							continue;
						}
						index.get(phrs, true).addAll(poss);
					}
					newIndex.clear();

					writePhrases(index, out);
				}
				out.close();

				int prog = BatchUtils.progress(file_loc + 1, files.size());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, file_loc + 1, files.size(), timer.stop());
				}
			}
			return null;
		}

		private ListMap<String, Long> createLocalIndex(File file) throws Exception {
			FileChannel fc = FileUtils.openFileChannel(file, "rw");
			ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc);
			fc.close();

			int i = 0;
			IntegerArray ws = ByteArrayUtils.toIntegerArray(data.get(i++));
			ListMap<String, Long> ret = Generics.newListMap(ws.size());

			for (int j = 0; j < ws.size(); j++) {
				int w = ws.get(j);
				String word = vocab.getObject(w);
				LongArray poss = ByteArrayUtils.toLongArray(data.get(i++));
				List<Long> l = Generics.newArrayList(poss.size());
				for (long pos : poss) {
					l.add(pos);
				}
				ret.put(word, l);
			}
			return ret;
		}

		private boolean filterPhrase(String phrs) {
			List<String> words = StrUtils.split("_", phrs);
			IntegerArray ws = new IntegerArray(words.size());

			{
				for (String word : words) {
					int w = vocab.indexOf(word);
					if (w >= 0) {
						ws.add(w);
					}
				}

				if (words.size() != ws.size()) {
					return true;
				}
			}

			{
				int w1 = ws.get(0);
				int w2 = ws.get(ws.size() - 1);

				String word_at_start = vocab.get(w1);
				String word_at_end = vocab.getObject(w2);

				if (stopwords.contains(w2) || pat1.matcher(word_at_end).find()) {
					return true;
				}

				if (pat2.matcher(phrs).find()) {
					return true;
				}
			}

			{
				int stopword_cnt = 0;
				for (int w : ws) {
					if (stopwords.contains(w)) {
						stopword_cnt++;
					}
				}
				if (stopword_cnt == ws.size()) {
					return true;
				}
			}

			{
				int cnt = 0;
				for (int i = 0; i < phrs.length(); i++) {
					char ch = phrs.charAt(i);
					Character.UnicodeBlock ub = Character.UnicodeBlock.of(ch);
					if (ub != UnicodeBlock.BASIC_LATIN) {
						cnt++;
					}
				}
				if (cnt > 0) {
					return true;
				}
			}

			{
				int[] open_cnts = new int[3];
				int[] close_cnts = new int[3];

				for (String word : words) {
					if (word.equals("(")) {
						open_cnts[0]++;
					} else if (word.equals(")")) {
						close_cnts[0]++;
					} else if (word.equals("[")) {
						open_cnts[1]++;
					} else if (word.equals("]")) {
						close_cnts[1]++;
					} else if (word.equals("{")) {
						open_cnts[2]++;
					} else if (word.equals("}")) {
						close_cnts[2]++;
					}
				}

				boolean has_pairs = true;
				for (int j = 0; j < open_cnts.length; j++) {
					if (open_cnts[j] != close_cnts[j]) {
						has_pairs = false;
						// System.out.println(phrs);
						break;
					}
				}

				if (!has_pairs) {
					return true;
				}
			}

			return false;
		}

		@Override
		protected void finalize() throws Throwable {
			super.finalize();
			dc.close();
		}

		private ListMap<Integer, Integer> getDocToLocs(ListMap<String, Long> index) {
			Counter<Integer> c = Generics.newCounter();
			for (String phrs : index.keySet()) {
				List<Long> poss = index.get(phrs);

				for (int k = 0; k < poss.size(); k++) {
					long pos = poss.get(k);
					long[] two = getTwoIndexes(pos);
					int dseq = (int) two[0];
					c.incrementCount(dseq, 1);
				}
			}

			ListMap<Integer, Integer> ret = Generics.newListMap(c.size());

			for (Entry<Integer, Double> e : c.entrySet()) {
				ret.put(e.getKey(), Generics.newArrayList(e.getValue().intValue()));
			}

			for (String phrs : index.keySet()) {
				List<Long> poss = index.get(phrs);

				for (int k = 0; k < poss.size(); k++) {
					long pos = poss.get(k);
					long[] two = getTwoIndexes(pos);
					int dseq = (int) two[0];
					int wloc = (int) two[1];
					ret.put(dseq, wloc);
				}
			}
			return ret;
		}

		private ListMap<String, Long> getNewIndex(ListMap<Integer, Integer> docToLocs, int phrs_len, int index_size) throws Exception {
			ListMap<String, Long> ret = Generics.newListMap(index_size);
			List<List<Integer>> ranges = getSequenceRanges(docToLocs);

			for (int i = 0; i < ranges.size(); i++) {
				List<Integer> range = ranges.get(i);
				List<Pair<String, IntegerArray>> ds = dc.getRange(range.get(0), range.get(1));

				for (int j = 0; j < ds.size(); j++) {
					int dseq = j + range.get(0);
					IntegerArray d = ds.get(j).getSecond();

					if (d.size() > max_doc_len) {
						d = d.subArray(0, max_doc_len);
					}

					List<Integer> starts = docToLocs.get(dseq);
					Collections.sort(starts);

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
						ret.put(phrs, pos);
					}
				}
			}

			return ret;
		}

		private List<List<Integer>> getSequenceRanges(ListMap<Integer, Integer> docToLocs) {
			IntegerArray dseqs = new IntegerArray(docToLocs.keySet());
			dseqs.sort(false);

			List<List<Integer>> ret = Generics.newArrayList(dseqs.size());

			int start = dseqs.get(0);
			int i = 1;

			while (i < dseqs.size()) {
				int prev = dseqs.get(i - 1);
				int cur = dseqs.get(i);

				if (cur - prev != 1) {
					int end = prev + 1;
					List<Integer> range = Generics.newArrayList(2);
					range.add(start);
					range.add(end);
					ret.add(range);
					start = cur;
				}
				i++;
			}

			int end = dseqs.get(dseqs.size() - 1) + 1;
			List<Integer> range = Generics.newArrayList(2);
			range.add(start);
			range.add(end);
			ret.add(range);
			// System.out.printf("[%d -> %d]\n", dseqs.size(), ret.size());
			return ret;
		}

		private void writePhrases(ListMap<String, Long> index, FileChannel out) throws Exception {
			List<String> phrss = Generics.newArrayList(index.keySet());
			Collections.sort(phrss);

			for (String phrs : phrss) {
				
				if (filterPhrase(phrs)) {
					continue;
				}

				List<Long> poss = index.get(phrs);
				Counter<Integer> c = Generics.newCounter();

				for (long pos : poss) {
					long[] two = getTwoIndexes(pos);
					int dseq = (int) two[0];
					c.incrementCount(dseq, 1);
				}

				ListMap<Integer, Integer> docToLocs = Generics.newListMap(c.size());

				for (Entry<Integer, Double> e : c.entrySet()) {
					ArrayList<Integer> l = (ArrayList<Integer>) docToLocs.ensure(e.getKey());
					l.ensureCapacity(e.getValue().intValue());
				}

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

				SPostingList pl = new SPostingList(phrs, dseqs, locData);
				SPostingList.writePostingList(pl, out, false);
			}
		}

	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		{

			Set<String> stopwords = FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);

			String[] dirs = { MIRPath.OHSUMED_COL_DC_DIR, MIRPath.TREC_CDS_2016_COL_DC_DIR, MIRPath.BIOASQ_COL_DC_DIR,
					MIRPath.WIKI_COL_DIR };

			for (int i = 0; i < dirs.length; i++) {
				if (i != 1) {
					continue;
				}

				String dir = dirs[i];
				FrequentPhraseDetector fpd = new FrequentPhraseDetector(dir, 6, 30, stopwords);
				fpd.setOutputFileSize(500);
				fpd.setThreadSize(1);
				fpd.detect(false);
			}
		}

		System.out.println("process ends.");
	}

	private DocumentCollection dc;

	private Vocab vocab;

	private Set<Integer> stopwords;

	private Pattern pat1 = Pattern.compile("^\\p{Punct}+");

	private Pattern pat2 = Pattern.compile("^[\\p{Punct}\\s]+");

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

	public FrequentPhraseDetector(String dataDir, int max_phrs_len, int min_support, Set<String> stopwords) throws Exception {
		this.dataDir = new File(dataDir);
		this.dc = new DocumentCollection(dataDir);
		this.max_phrs_len = max_phrs_len;
		this.min_support = min_support;
		this.stopwords = Generics.newHashSet();

		if (stopwords != null && stopwords.size() > 0) {
			for (String word : stopwords) {
				int w = dc.getVocab().indexOf(word);
				if (w >= 0) {
					this.stopwords.add(w);
				}
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

		ListList<Integer> wsData = Generics.newListList(output_file_size);
		ListList<Long> posData = Generics.newListList(output_file_size);

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

					if (w == DocumentCollection.SENT_END || vocab.getCount(w) < min_support) {
						continue;
					}

					// if (w == DocumentCollection.SENT_END || stopwords.contains(w) || vocab.getCount(w) < min_support) {
					// continue;
					// }

					String word = vocab.getObject(w);

					if (pat1.matcher(word).find()) {
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
		ListMap<Integer, Long> lm1 = Generics.newListMap();
		ListMap<Integer, Long> lm2 = Generics.newListMap();

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);

			{
				FileChannel fc = FileUtils.openFileChannel(file, "r");

				while (fc.position() < fc.size()) {
					ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc);
					IntegerArray ws = ByteArrayUtils.toIntegerArray(data.get(0));
					LongArray poss = ByteArrayUtils.toLongArray(data.get(1));

					for (int j = 0; j < ws.size(); j++) {
						int w = ws.get(j);
						long pos = poss.get(j);
						lm2.put(w, pos);
					}

					for (int w : lm2.keySet()) {
						lm1.get(w, true).addAll(lm2.get(w));
					}
					lm2.clear();
				}
				fc.close();
			}

			{
				IntegerArray ws = new IntegerArray(lm1.keySet());
				ws.sort(false);

				ByteArrayMatrix data = new ByteArrayMatrix(ws.size() + 1);
				data.add(ByteArrayUtils.toByteArray(ws));

				for (int w : ws) {
					LongArray poss = new LongArray(lm1.get(w));
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
			lm1.clear();
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
