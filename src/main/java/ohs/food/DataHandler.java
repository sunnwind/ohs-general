package ohs.food;

import java.io.File;
import java.util.List;

import kr.co.shineware.nlp.komoran.core.analyzer.Komoran;
import kr.co.shineware.util.common.model.Pair;
import ohs.io.FileUtils;
import ohs.types.generic.Counter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.UnicodeUtils;

public class DataHandler {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// filter();

		tagPOS();

		System.out.println("process ends.");
	}

	public static void tagPOS() throws Exception {
		Komoran komoran = new Komoran("lib/models-full/");

		List<File> files = FileUtils.getFilesUnder("../../data/food_data/식품관련학술지_선별_텍스트_필터");

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);
			String text = FileUtils.readFromText(file);

			String[] lines = text.split("[\n]+");
			StringBuffer sb = new StringBuffer();
			for (String line : lines) {
				line = line.trim();
				String t = getText(komoran.analyze(line, 1));
				sb.append(t + "\n");
			}

			String text2 = sb.toString().trim();

			String outFileName = file.getPath().replace("텍스트_필터", "텍스트_필터_pos");

			FileUtils.writeAsText(outFileName, text2);

		}

	}

	private static String getText(List<List<List<Pair<String, String>>>> result) {
		StringBuffer sb = new StringBuffer();

		for (int i = 0; i < result.size(); i++) {
			List<List<Pair<String, String>>> ll = result.get(i);
			for (int j = 0; j < ll.size(); j++) {
				List<Pair<String, String>> l = ll.get(j);

				for (int k = 0; k < l.size(); k++) {
					Pair<String, String> pair = l.get(k);
					String f = pair.getFirst().replace(" ", "_");
					String s = pair.getSecond();

					if (s.length() == 0) {
						continue;
					}

					sb.append(String.format("%s%s%s", f, "/", s));

					if (k != l.size() - 1) {
						sb.append(" ");
					}
				}

				if (j != ll.size() - 1) {
					sb.append(" ");
				}
			}
			if (i != ll.size() - 1) {
				sb.append("\n");
			}
		}

		return sb.toString().trim();
	}

	public static void filter() throws Exception {
		String dirPath = "G:/data/food data/식품관련학술지_선별_텍스트/";

		List<File> files = FileUtils.getFilesUnder(dirPath);

		FileUtils.deleteFilesUnder(files.get(0).getPath().replace("식품관련학술지_선별_텍스트", "식품관련학술지_선별_텍스트_필터"));

		for (int i = 0; i < files.size(); i++) {
			String inFileName = files.get(i).getPath();
			// if (!files.get(i).getPath().contains("v31n6_781.txt")) {
			// continue;
			// }
			String text = FileUtils.readFromText(inFileName, "euc-kr");
			// System.out.println(text);

			Counter<Character> c = Generics.newCounter();
			Counter<String> c2 = Generics.newCounter();

			for (String word : StrUtils.split(text)) {
				for (int j = 0; j < word.length(); j++) {
					char ch = word.charAt(j);
					if (UnicodeUtils.isKorean(ch)) {
						c2.incrementCount("KOR", 1);
					} else {
						c2.incrementCount("ELSE", 1);
					}
					c.incrementCount(word.charAt(j), 1);
				}
			}

			// System.out.println("--------------------");
			// System.out.println(inFileName);
			// System.out.println("--------------------");
			// System.out.println(c.toString());
			// System.out.println("--------------------");

			// List<Character> keys = c.getSortedKeys();
			// boolean toFilter = false;
			// for (int j = 0; j < 5 && j < keys.size(); j++) {
			//
			// if (keys.get(j) == '?') {
			// toFilter = true;
			// break;
			// }
			// }

			double kor_cnt = c2.getCount("KOR") + 1;
			double other_cnt = c2.getCount("ELSE") + 1;

			double ratio = kor_cnt / other_cnt;

			if (ratio > 1.5) {
				String outFileName = inFileName.replace("식품관련학술지_선별_텍스트", "식품관련학술지_선별_텍스트_필터");
				FileUtils.writeAsText(outFileName, text);
			}
		}
	}

}
