package ohs.corpus.dump;

import java.io.File;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import ohs.io.FileUtils;
import ohs.io.TextFileWriter;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.general.NLPUtils;
import ohs.types.generic.SetMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class TokenizeHandler {

	class Worker implements Callable<Integer> {

		public Worker() {
			super();
		}

		@Override
		public Integer call() throws Exception {

			int loc = 0;
			while ((loc = file_cnt.getAndIncrement()) < inFiles.size()) {
				String inFileName = inFiles.get(loc).getPath().replace("\\", "/");
				String outFileName = inFileName.replace(inDir, outDir);
				String logFileName = inFileName.replace(inDir, logDir).replace(".txt.gz", ".txt");

				if (FileUtils.exists(outFileName)) {
					continue;
				}

				List<String> lines = FileUtils.readLinesFromText(inFileName);

				SetMap<Integer, String> sm = Generics.newSetMap(lines.size());

				if (FileUtils.exists(logFileName)) {
					for (String line : FileUtils.readLinesFromText(logFileName)) {
						String[] ps = line.split("\t");
						int i = Integer.parseInt(ps[0]);
						sm.put(i, ps[1]);
					}
				}

				boolean[] skips = new boolean[lines.size()];

				for (int i : sm.keySet()) {
					Set<String> s = sm.get(i);
					if (s.contains("START") && !s.contains("END")) {
						skips[i] = true;
					}
				}

				TextFileWriter writer = new TextFileWriter(logFileName);

				for (int i = 0; i < lines.size(); i++) {
					writer.write(String.format("%d\t%s\n", i, "START"));

					if (skips[i]) {
						continue;
					}

					String line = lines.get(i);
					String[] parts = line.split("\t");
					try {
						parts = StrUtils.unwrap(parts);

						for (int dataLoc : dataLocs) {
							String s = parts[dataLoc];
							StringBuffer sb = new StringBuffer();

							for (String t : s.split(StrUtils.LINE_REP)) {
								if (t.length() == 0) {
									sb.append("\n\n");
								} else {
									sb.append(StrUtils.join("\n", NLPUtils.tokenize(t)));
									sb.append("\n");
								}
							}
							s = sb.toString().trim();
							s = s.replace("\n", StrUtils.LINE_REP);
							s = StrUtils.normalizeSpaces(s);
							parts[dataLoc] = s;
						}
						parts = StrUtils.wrap(parts);
						lines.set(i, StrUtils.join("\t", parts));
					} catch (Exception e) {
						e.printStackTrace();
					}

					writer.write(String.format("%d\t%s", i, "END"));

					if (i != lines.size() - 1) {
						writer.write("\n");
					}
				}
				writer.close();

				FileUtils.writeStringCollectionAsText(outFileName, lines);
			}

			return (int) 0;
		}
	}

	// ptb3Escaping=false,normalizeParentheses=false,normalizeOtherBrackets=false

	public static TokenizerFactory<? extends HasWord> tf = PTBTokenizer.factory(new CoreLabelTokenFactory(),
			"ptb3Escaping=false,normalizeParentheses=false,normalizeOtherBrackets=false");

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		TokenizeHandler th = new TokenizeHandler();
		th.setThreadSize(10);

		// th.tokenize(MIRPath.OHSUMED_COL_LINE_DIR, new int[] { 3, 5 }, MIRPath.OHSUMED_COL_TOK_DIR, false);
		// th.tokenize(MIRPath.CLEF_EH_2014_COL_LINE_DIR, new int[] { 3 },
		// MIRPath.CLEF_EH_2014_COL_TOK_DIR, true);
		// th.tokenize(MIRPath.TREC_GENO_2007_COL_LINE_DIR, new int[] { 1 },
		// MIRPath.TREC_GENO_2007_COL_TOK_DIR, false);
		// th.tokenize(MIRPath.TREC_CDS_2014_COL_LINE_DIR, new int[] { 1, 2, 3, 4 }, MIRPath.TREC_CDS_2014_COL_TOK_DIR, true);
		// th.tokenize(MIRPath.TREC_CDS_2016_COL_LINE_DIR, new int[] { 1, 2, 3, 4 }, MIRPath.TREC_CDS_2016_COL_TOK_DIR, true);
		// th.tokenize(MIRPath.WIKI_COL_LINE_DIR, new int[] { 3 }, MIRPath.WIKI_COL_TOK_DIR, true);
		// th.tokenize(MIRPath.CLUEWEB_COL_LINE_DIR, new int[] { 1 }, MIRPath.CLUEWEB_COL_TOK_DIR, false);
		// th.tokenize(MIRPath.BIOASQ_COL_LINE_DIR, new int[] { 4, 5 }, MIRPath.BIOASQ_COL_TOK_DIR, false);
		// th.tokenize("../../data/medical_ir/scopus/col/line/", new int[] { 1, 2, 3 }, "../../data/medical_ir/scopus/col/tok/", true);

		th.tokenize(MIRPath.TREC_PM_2017_COL_MEDLINE_LINE_DIR, new int[] { 2, 3, 4 }, MIRPath.TREC_PM_2017_COL_MEDLINE_TOK_DIR, true);
		th.tokenize(MIRPath.TREC_PM_2017_COL_CLINICAL_LINE_DIR, new int[] { 1, 2, 2, 3, 4, 5 }, MIRPath.TREC_PM_2017_COL_CLINICAL_TOK_DIR,
				true);

		System.out.println("process ends.");
	}

	private int[] dataLocs;

	private AtomicInteger file_cnt;

	private String inDir;

	private String outDir;

	private List<File> inFiles;

	private String logDir;

	public TokenizeHandler() {
	}

	public void tokenize() throws Exception {

	}

	private int thread_size = 10;

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

	public void tokenize(String inDir, int[] dataLocs, String outDir, boolean delete) throws Exception {
		System.out.printf("tokenize [%s] to [%s].\n", inDir, outDir);

		this.inDir = inDir;
		this.dataLocs = dataLocs;
		this.outDir = outDir;
		logDir = inDir.replace(inDir, outDir).replace("tok", "tok_log");

		if (delete) {
			FileUtils.deleteFilesUnder(outDir);
			FileUtils.deleteFilesUnder(logDir);
		}

		inFiles = FileUtils.getFilesUnder(inDir);

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList();

		file_cnt = new AtomicInteger(0);

		for (int i = 0; i < inFiles.size() && i < thread_size; i++) {
			fs.add(tpe.submit(new Worker()));
		}

		for (int i = 0; i < fs.size(); i++) {
			fs.get(i).get();
		}

		tpe.shutdown();
	}

}
