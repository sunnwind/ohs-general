package ohs.disaster;

import java.io.File;
import java.lang.Character.UnicodeBlock;
import java.util.List;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class DataHandler {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// IntegerArray a = new IntegerArray(new int[] { 1, 2, 3, 4, 5 });
		//
		// System.out.println(a);
		//
		// for (int v : a) {
		// System.out.println(v);
		// }

		DataHandler dh = new DataHandler();
		dh.process1();
		// dh.process2();

		System.out.println("process ends.");

	}

	public List<String> getDocument(TextFileReader reader) {
		List<String> ret = Generics.newArrayList();
		while (reader.hasNext()) {
			String line = reader.next();
			if (line.startsWith("<doc>") || line.startsWith("file_name:")) {

			} else if (line.startsWith("</doc>")) {
				break;
			} else {
				ret.add(line);
			}
		}
		return ret;
	}

	public void process1() throws Exception {
		// TextFileWriter writer = new TextFileWriter(DSTPath.DATA_DIR + "daily_content.txt");

		List<File> files = FileUtils.getFilesUnder(DSTPath.COL_RAW_DIR);

		CounterMap<String, String> cm1 = Generics.newCounterMap();
		CounterMap<String, String> cm2 = Generics.newCounterMap();
		CounterMap<String, String> cm3 = Generics.newCounterMap();
		Counter<String> cc = Generics.newCounter();

		for (File file : files) {
			// if (!file.getName().contains("고니")) {
			// continue;
			// }
			String text = FileUtils.readFromText(file);

			StringBuffer sb = new StringBuffer();
			sb.append("<doc>");
			sb.append(String.format("\nfile_name:\t%s", file.getName()));
			sb.append(text);
			sb.append("\n</doc>");

			// writer.write(sb.toString() + "\n\n");

			String[] lines = text.split(StrUtils.LINE_REP);

			for (String line : lines) {
				List<UnicodeBlock> ubs = Generics.newArrayList();
				for (char c : line.toCharArray()) {
					UnicodeBlock ub = Character.UnicodeBlock.of(c);
					ubs.add(ub);

					if (ub == UnicodeBlock.HANGUL_SYLLABLES) {
						continue;
					}

					cm1.incrementCount(ub.toString(), c + "", 1);

					// System.out.printf("%s, %s\n", c, ub);
				}

				List<String> words = StrUtils.split("[\u2029\\s\\p{Punct}]+", line);

				for (int i = 0; i < words.size(); i++) {
					String word = words.get(i);
					cc.incrementCount(word, 1);

					if (i > 0) {
						cm2.incrementCount(words.get(i - 1), words.get(i), 1);
					}
				}
			}
		}

		FileUtils.writeStringCounterAsText(DSTPath.DATA_DIR + "unigrams.txt", cc);
		FileUtils.writeStringCounterMapAsText(DSTPath.DATA_DIR + "bigrams.txt", cm2);
		FileUtils.writeStringCounterMapAsText(DSTPath.DATA_DIR + "char.txt", cm1);

		// writer.close();
	}

	public void process2() {
		TextFileReader reader = new TextFileReader(DSTPath.DATA_DIR + "daily_content.txt");
		Counter<String> c = Generics.newCounter();

		while (reader.hasNext()) {
			List<String> lines = getDocument(reader);

			for (String line : lines) {
				int cnt = 0;

				for (int i = 0; i < line.length(); i++) {
					if (line.charAt(i) == ' ') {
						cnt++;
					} else {
						break;
					}
				}

				if (cnt == 0) {
					// System.out.println(line);
					c.incrementCount(line, 1);
				}
			}

			// System.out.println(StrUtils.join("\n", lines));
		}
		reader.close();

		System.out.println(c.toString());
	}

}
