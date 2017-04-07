package ohs.corpus.dump;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;

import ohs.io.FileUtils;
import ohs.utils.Generics;

public class DisasterDumper extends TextDumper {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		DisasterDumper dh = new DisasterDumper("../../data/disaster/col/raw/", "../../data/disaster/col/line/");
		dh.dump();
		System.out.println("process ends.");
	}

	public DisasterDumper(String inDirName, String outDirName) {
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

		List<String> res = Generics.newArrayList();

		int batch_cnt = 0;

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);
			String s = FileUtils.readFromText(file);
			String s2 = s.replace("\n", "<nl>");

			res.add(s2);

			if (res.size() % batch_size == 0) {
				DecimalFormat df = new DecimalFormat("000000");
				String outFileName = outPathName + String.format("%s.txt.gz", df.format(batch_cnt++));
				FileUtils.writeStringCollectionAsText(outFileName, res);
				res.clear();
			}
		}

		if (res.size() > 0) {
			DecimalFormat df = new DecimalFormat("000000");
			String outFileName = outPathName + String.format("%s.txt.gz", df.format(batch_cnt++));
			FileUtils.writeStringCollectionAsText(outFileName, res);
			res.clear();
		}
	}
}
