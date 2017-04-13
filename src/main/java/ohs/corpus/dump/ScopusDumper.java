package ohs.corpus.dump;

import java.text.DecimalFormat;
import java.util.List;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.types.generic.ListMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class ScopusDumper extends TextDumper {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		{
			ScopusDumper dh = new ScopusDumper("../../data/medical_ir/scopus/col/raw/", "../../data/medical_ir/scopus/col/line/");
			dh.dump();
		}

		System.out.println("process ends.");
	}

	public ScopusDumper(String inDirName, String outDirName) {
		super(inDirName, outDirName);
	}

	@Override
	public void dump() throws Exception {
		System.out.printf("dump [%s] to [%s].\n", inPathName, outPathName);

		ListMap<String, String> lm = Generics.newListMap(1000000);

		{
			TextFileReader reader = new TextFileReader(inPathName + "scopus_keyword.csv.gz");
			while (reader.hasNext()) {
				String line = reader.next();

				if (reader.getLineCnt() == 1) {
					continue;
				}

				try {
					String[] ps = line.split("\t");
					ps = StrUtils.unwrap(ps);
					if (ps.length == 2) {
						lm.put(ps[0], ps[1]);
					} else {
//						System.out.println(line);
					} 
				} catch (Exception e) {
					System.out.println(line);
				}
			}
			reader.close();
		}

		{
			TextFileReader reader = new TextFileReader(inPathName + "scopus_document.csv.gz");
			List<String> lines = Generics.newArrayList();
			int cnt = 0;

			while (reader.hasNext()) {
				String line = reader.next();
				if (reader.getLineCnt() == 1) {
					continue;
				}
				if (lines.size() > 0 && lines.size() % batch_size == 0) {
					DecimalFormat df = new DecimalFormat("00000");
					String outFileName = String.format("%s/%s.txt.gz", outPathName, df.format(cnt++));
					FileUtils.writeStringCollectionAsText(outFileName, lines);
					lines.clear();
				} else {
					List<String> ps = StrUtils.split("\t", line);
					ps = StrUtils.unwrap(ps);

					List<String> l = lm.get(ps.get(0), false);

					if (l != null) {
						ps.add(StrUtils.join(StrUtils.LINE_REP, l));
						ps = StrUtils.wrap(ps);
						lines.add(StrUtils.join("\t", ps));
					}
				}
			}
			reader.close();

			DecimalFormat df = new DecimalFormat("00000");
			String outFileName = String.format("%s/%s.txt.gz", outPathName, df.format(cnt++));
			FileUtils.writeStringCollectionAsText(outFileName, lines);
		}

	}

}
