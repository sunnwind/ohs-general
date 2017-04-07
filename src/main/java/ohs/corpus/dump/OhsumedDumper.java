package ohs.corpus.dump;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.ir.medical.general.MIRPath;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class OhsumedDumper extends TextDumper {

	class Worker implements Callable<Integer> {

		private String inFileName;

		public Worker(String inFileName) {
			super();
			this.inFileName = inFileName;
		}

		@Override
		public Integer call() throws Exception {

			String outPath = new File(inFileName.replace(inPathName, outPathName)).getParent().replace("\\", "/");

			TextFileReader reader = new TextFileReader(inFileName);

			Map<String, String> map = Generics.newHashMap();
			List<String> res = Generics.newArrayList();

			while (reader.hasNext()) {
				String line = reader.next();

				if (line.startsWith(".I")) {
					String[] parts = line.split(" ");
					String key = parts[0];
					String value = parts[1];

					if (map.size() == 0) {
						map.put(key, value);
					} else {
						res.add(makeOutput(map));

						if (res.size() % batch_size == 0) {
							DecimalFormat df = new DecimalFormat("00000");
							String outFileName = String.format("%s/%s.txt.gz", outPath, df.format(batch_cnt.getAndIncrement()));
							FileUtils.writeStringCollectionAsText(outFileName, res);
							res.clear();
						}

						map.clear();
						map.put(key, value);
					}
				} else {
					reader.hasNext();
					String value = reader.next();
					map.put(line, value);
				}
			}
			reader.close();

			if (res.size() > 0) {
				DecimalFormat df = new DecimalFormat("00000");
				String outFileName = String.format("%s/%s.txt.gz", outPath, df.format(batch_cnt.getAndIncrement()));
				FileUtils.writeStringCollectionAsText(outFileName, res);
			}

			return (int) reader.getLineCnt();
		}

		private String makeOutput(Map<String, String> map) {
			String seqId = map.get(".I");
			String medlineId = map.get(".U");
			String meshTerms = map.get(".M");
			String title = map.get(".T");
			String publicationType = map.get(".P");
			String abs = map.get(".W");
			String authors = map.get(".A");
			String source = map.get(".S");

			String[] values = new String[] { seqId, medlineId, meshTerms, title, publicationType, abs, authors, source };

			for (int i = 0; i < values.length; i++) {
				if (values[i] == null) {
					values[i] = "";
				}
				values[i] = StrUtils.normalizeSpaces(values[i]);
			}

			values = StrUtils.wrap(values);
			String output = StrUtils.join("\t", values);
			return output;
		}

	}

	public static final String[] TAGS = { ".I", ".U", ".M", ".T", ".P", ".W", ".A", ".S" };

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		OhsumedDumper dh = new OhsumedDumper(MIRPath.OHSUMED_COL_RAW_DIR, MIRPath.OHSUMED_COL_LINE_DIR);
		dh.dump();
		System.out.println("process ends.");
	}

	private AtomicInteger batch_cnt;

	public OhsumedDumper(String inputDir, String outputFileName) {
		super(inputDir, outputFileName);
	}

	@Override
	public void dump() throws Exception {
		System.out.printf("dump [%s] to [%s]\n", inPathName, outPathName);
		Timer timer = new Timer();
		timer.start();

		FileUtils.deleteFilesUnder(outPathName);

		/*
		 * .I sequential identifier
		 * 
		 * .U MEDLINE identifier (UI)
		 * 
		 * .M Human-assigned MeSH terms (MH)
		 * 
		 * .T Title (TI)
		 * 
		 * .P Publication type (PT)
		 * 
		 * .W Abstract (AB)
		 * 
		 * .A Author (AU)
		 * 
		 * .S Source (SO)
		 */

		File[] files = new File(inPathName).listFiles();

		int thread_size = files.length;

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList();

		batch_cnt = new AtomicInteger(0);

		for (int i = 0; i < files.length; i++) {
			String inFileName = files[i].getPath().replace("\\", "/");
			fs.add(tpe.submit(new Worker(inFileName)));
		}

		for (int i = 0; i < fs.size(); i++) {
			fs.get(i).get();
		}

		tpe.shutdown();
	}

}
