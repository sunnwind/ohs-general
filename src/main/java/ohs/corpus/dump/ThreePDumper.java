package ohs.corpus.dump;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;

import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class ThreePDumper extends TextDumper {

	public static List<String> getKeywords(String keywordStr) {
		List<String> ret = Generics.newArrayList();
		for (String kw : keywordStr.split(";")) {
			kw = kw.trim();
			if (kw.length() > 0) {
				ret.add(kw);
			}
		}
		return ret;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		ThreePDumper dh = new ThreePDumper(KPPath.COL_DB_DIR, KPPath.COL_LINE_DIR);
		dh.dump();
		System.out.println("process ends.");
	}

	public ThreePDumper(String inDirName, String outDirName) {
		super(inDirName, outDirName);
	}

	@Override
	public void dump() throws Exception {
		System.out.printf("dump [%s] to [%s].\n", inPathName, outPathName);

		List<File> files = FileUtils.getFilesUnder(inPathName);
		// String labelStr =
		// "TYPE\tCN\tKOR_KWD\tENG_KWD\tENG_TITLE\tKOR_TITLE\tKOR_ABS\tENG_ABS";
		int batch_size = 100000;

		FileUtils.deleteFilesUnder(outPathName);

		for (int i = 0; i < files.size(); i++) {
			File inFile = files.get(i);
			String fileName = inFile.getName();
			String type = "";

			if (!fileName.contains("_2")) {
				continue;
			}

			if (fileName.contains("paper")) {
				type = "paper";
			} else if (fileName.contains("report")) {
				type = "report";
			} else if (fileName.contains("patent")) {
				type = "patent";
			}

			// if (!type.equalsIgnoreCase("patent")) {
			// continue;
			// }

			// TextFileWriter writer = new
			// TextFileWriter(KPPath.SINGLE_DUMP_FILE);
			// writer.write();

			// String[] inFileNames = { KPPath.PAPER_DUMP_FILE,
			// KPPath.REPORT_DUMP_FILE, KPPath.PATENT_DUMP_FILE };

			List<String> res = Generics.newArrayList();
			// res.add(labelStr);

			int batch_cnt = 0;

			TextFileReader reader = new TextFileReader(inFile.getPath());
			List<String> labels = Generics.newArrayList();

			while (reader.hasNext()) {
				String line = reader.next();
				String[] ps = line.split("\t");

				// System.out.println(line);

				if (reader.getLineCnt() == 1) {
					ps = StrUtils.unwrap(ps);

					for (String p : ps) {
						labels.add(p);
					}
				} else {
					if (ps.length != labels.size()) {
						System.out.println(line);
						continue;
					}

					ps = StrUtils.unwrap(ps);

					String cn = ps[0];
					String korTitle = "";
					String engTitle = "";
					String korAbs = "";
					String engAbs = "";
					String korKwdStr = "";
					String engKwdStr = "";
					String date = "";

					if (type.equals("paper")) {
						korTitle = ps[1];
						engTitle = ps[2];
						korAbs = ps[5];
						engAbs = ps[6];
						korKwdStr = ps[3];
						engKwdStr = ps[4];
						date = ps[7];
					} else if (type.equals("report")) {
						korTitle = ps[1];
						engTitle = ps[2];
						korAbs = ps[5];
						engAbs = ps[6];
						korKwdStr = ps[3];
						engKwdStr = ps[4];
						date = ps[7];
					} else if (type.equals("patent")) {
						String applno = ps[0];
						korTitle = ps[1];
						engTitle = ps[2];
						cn = ps[3];
						korAbs = ps[4];
						date = ps[5];
					}

					List<String> korKwds = getKeywords(korKwdStr);
					List<String> engKwds = getKeywords(engKwdStr);

					ps = new String[] { type, cn, StrUtils.join(StrUtils.LINE_REP, korKwds),
							StrUtils.join(StrUtils.LINE_REP, engKwds), korTitle, engTitle, korAbs, engAbs, date };

					res.add(StrUtils.join("\t", StrUtils.wrap(ps)));

					if (res.size() % batch_size == 0) {
						DecimalFormat df = new DecimalFormat("000000");
						String outFileName = outPathName + String.format("%s/%s.txt.gz", type, df.format(batch_cnt++));
						FileUtils.writeStringCollectionAsText(outFileName, res);

						res.clear();
						// res.add(labelStr);
					}
				}
			}
			reader.close();

			if (res.size() > 0) {
				DecimalFormat df = new DecimalFormat("000000");
				String outFileName = outPathName + String.format("%s/%s.txt.gz", type, df.format(batch_cnt++));
				FileUtils.writeStringCollectionAsText(outFileName, res);

				res.clear();
				// res.add(labelStr);
			}
		}
	}
}
