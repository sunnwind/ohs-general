package ohs.food;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.io.TextFileWriter;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class Ranker {

	public static Counter<String> getQueryWordCnts() {
		List<String> words = Generics.newArrayList();
		words.add("식품/NNG");
		words.add("음식/NNG");
		words.add("식재료/NNG");
		words.add("먹거리/NNG");
		words.add("먹/VV");
		words.add("요리/NNG");
		words.add("약품/NNG");
		words.add("의약품/NNG");
		words.add("영양소/NNG");
		words.add("과일/NNG");
		words.add("곡류/NNG");
		words.add("육류/NNG");
		words.add("어류/NNG");
		words.add("어패류/NNG");
		words.add("유제품/NNG");
		words.add("농산물/NNG");
		words.add("정크푸드/NNG");
		words.add("조미료/NNG");
		words.add("음료/NNG");
		words.add("농수산물/NNG");

		Counter<String> queryWordCnts = Generics.newCounter();

		for (String word : words) {
			queryWordCnts.incrementCount(word, 1);
		}

		return queryWordCnts;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Map<String, String> map = Generics.newHashMap();
		String outFileName = "";
		boolean isDonga = false;

		if (isDonga) {
			TextFileReader reader = new TextFileReader(FoodPath.DONGA_SCIENCE_FILE, "euc-kr");
			while (reader.hasNext()) {
				List<String> lines = reader.nextLines();

				if (lines.size() == 0) {
					continue;
				}

				String id = lines.get(0).split("/")[0];
				id = "donga-" + id;
				map.put(id, StrUtils.join("\n", lines));

				System.out.println(id);
			}
			reader.close();

			outFileName = FoodPath.DONGA_SCIENCE_RANKED_FILE;
		} else {
			List<File> files = FileUtils.getFilesUnder("../../data/food_data/식품관련학술지_선별_텍스트_필터_pos");

			for (int i = 0; i < files.size(); i++) {
				File file = files.get(i);
				String id = file.getPath();
				int loc = id.indexOf("식품관련학술지_선별_텍스트_필터_pos");
				id = id.substring(loc + "식품관련학술지_선별_텍스트_필터_pos".length() + 1);
				id = "paper-" + id;
				String text = id + "\n" + FileUtils.readFromText(file);
				map.put(id, text);
			}

			outFileName = "../../data/food_data/식품관련학술지_선별_텍스트_필터_pos_ranked.txt";
		}

		rank(getQueryWordCnts(), map, outFileName);

		System.out.println("process ends.");
	}

	public static void rank(Counter<String> queryWordCnts, Map<String, String> map, String outFileName) throws Exception {
		Vocab vocab = new Vocab();

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (String id : map.keySet()) {
			List<String> lines = Arrays.asList(map.get(id).split("\n"));

			for (int i = 1; i < lines.size(); i++) {
				String[] parts = lines.get(i).split(" ");
				for (int j = 0; j < parts.length; j++) {
					String s = parts[j];
					// String[] two = StrUtils.split2Two("/", s);

					// String word = two[0];
					// String pos = two[1];
					cm.incrementCount(id, s, 1);
					// cm.incrementCount(id, word, 1);
					// cm.incrementCount(id, pos, 1);
				}
			}
		}

		Map<String, SparseVector> m = Generics.newHashMap();

		for (String id : cm.keySet()) {
			Counter<String> c = cm.getCounter(id);
			m.put(id, VectorUtils.toSparseVector(c, vocab, true));
		}

		TermWeighting.bm25(m.values());

		SparseVector q = VectorUtils.toSparseVector(queryWordCnts, vocab, true);

		Counter<String> docScores = Generics.newCounter();

		for (String id : m.keySet()) {
			SparseVector wordWeights = m.get(id);
			double score = VectorMath.dotProduct(q, wordWeights);
			docScores.setCount(id, Math.exp(score));
		}

		TextFileWriter writer = new TextFileWriter(outFileName);

		List<String> ids = docScores.getSortedKeys();

		for (int i = 0; i < ids.size() && i < 3000; i++) {
			String id = ids.get(i);
			String text = "";
			if (id.startsWith("paper-")) {
				String dataFileName = "../../data/food_data/식품관련학술지_선별_텍스트_필터/" + id.substring(6);
				text = FileUtils.readFromText(dataFileName);
				writer.write(id + "\n" + text);
			} else {
				text = map.get(id);
				writer.write(text);
			}

			if (i != ids.size() - 1) {
				writer.write("\n\n");
			}
		}
		writer.close();
	}

}
