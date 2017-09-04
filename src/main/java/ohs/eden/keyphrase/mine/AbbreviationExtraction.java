package ohs.eden.keyphrase.mine;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ohs.corpus.type.EnglishTokenizer;
import ohs.eden.keyphrase.cluster.AbbreviationExtractor;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.general.NLPUtils;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Pair;
import ohs.types.generic.SetMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class AbbreviationExtraction {

	public static void extract() throws Exception {

		AbbreviationExtractor ext = new AbbreviationExtractor();

		List<String> phrss = Generics.newLinkedList();

		for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_merged.txt")) {
			String[] ps = line.split("\t");
			phrss.add(ps[0]);
		}

		phrss = Generics.newArrayList(phrss);

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (int i = 0; i < phrss.size(); i++) {
			String phrs = phrss.get(i);
			phrs = StrUtils.join("\n", NLPUtils.tokenize(phrs));
			// content = content.replace("( ", "(").replace(" )", ")");

			List<Pair<String, String>> pairs = ext.extract(phrs);

			for (Pair<String, String> pair : pairs) {
				String shortForm = pair.getFirst();
				String longForm = pair.getSecond();
				cm.incrementCount(shortForm, longForm.toLowerCase(), 1);
			}
		}

		FileUtils.writeStringCounterMapAsText(MIRPath.PHRS_DIR + "abbr.txt", cm);
	}

	public static void extractCandidates() throws Exception {
		List<String> phrss = Generics.newLinkedList();

		for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_merged.txt")) {
			String[] ps = line.split("\t");
			String phrs = ps[0];

			List<String> words = StrUtils.split(phrs);
			boolean store = false;

			for (String word : words) {
				if (word.startsWith("(") && word.endsWith(")")) {
					store = true;
				}
			}

			if (store) {
				phrss.add(ps[0]);
			}
		}

		FileUtils.writeStringCollectionAsText(MIRPath.PHRS_DIR + "abbr_candidate.txt", phrss);
	}

	public static void filterCandidates() throws Exception {
		Pattern p = Pattern.compile("\\(([\\p{Alnum}]+)\\)");

		Set<String> set = Generics.newHashSet();

		EnglishTokenizer et = new EnglishTokenizer();

		for (String line : FileUtils.readLinesFromText(MIRPath.PHRS_DIR + "abbr_candidate.txt")) {
			String[] ps = line.split("\t");
			String phrs = ps[0];

			Matcher m = p.matcher(phrs);

			String longForm = "";
			String shortForm = "";

			if (!m.find()) {
				continue;
			}

			longForm = phrs.substring(0, m.start());
			shortForm = m.group(1);

			longForm = normalizeLongForm(longForm, shortForm);

			if (!isValid(longForm)) {
				continue;
			}

			if (!isValid(longForm, shortForm)) {
				continue;
			}

			// String s = StrUtils.join(" ", et.tokenize(longForm));

			set.add(String.format("%s\t%s", longForm.toLowerCase(), shortForm));
		}

		List<String> abbrs = Generics.newArrayList(set);

		Collections.sort(abbrs);

		FileUtils.writeStringCollectionAsText(MIRPath.PHRS_DIR + "abbr_filtered.txt", abbrs);
	}

	public static boolean isValid(String lf) {
		Pattern p = Pattern.compile("^\\p{Alpha}");
		boolean ret = p.matcher(lf).find() ? true : false;
		return ret;
	}

	public static boolean isValid(String lf, String sf) {
		lf = lf.toLowerCase();
		sf = sf.toLowerCase();

		boolean ret = false;

		int i = 0, j = 0;

		while (i < lf.length() && j < sf.length()) {
			char c1 = lf.charAt(i);
			char c2 = sf.charAt(j);

			if (c1 == c2) {
				i++;
				j++;
			} else {
				i++;
			}
		}

		if (sf.length() > 1 && sf.length() == j) {
			ret = true;
		}

		return ret;

	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		// extract();
		// extractCandidates();
		// filterCandidates();
		// tokenize();
		System.out.println("process ends.");
	}

	public static String normalizeLongForm(String lf, String sf) {
		List<String> words = StrUtils.split(lf);

		int ch = sf.toLowerCase().charAt(0);
		int j = 0;

		for (int i = 0; i < words.size(); i++) {
			String word = words.get(i).toLowerCase();
			if (word.indexOf(ch) >= 0) {
				j = i;
				break;
			}
		}

		if (j > 0) {
			lf = StrUtils.join(" ", words, j, words.size());
		}

		return lf;
	}

	public static void tokenize() throws Exception {
		EnglishTokenizer et = new EnglishTokenizer();

		Set<String> set = Generics.newHashSet();

		SetMap<String, String> sm = Generics.newSetMap();

		for (String line : FileUtils.readLinesFromText(MIRPath.PHRS_DIR + "abbr_filtered.txt")) {
			String[] ps = line.split("\t");
			String lf = ps[0];
			String sf = ps[1];

			lf = StrUtils.join(" ", et.tokenize(lf));
			sf = et.getStringNormalizer().normalize(sf);
			sm.put(sf, lf);
			set.add(String.format("%s\t%s", lf, sf));
		}

		// List<String> sfs = Generics.newArrayList(sm.keySet());
		// Collections.sort(sfs);

		FileUtils.writeStringSetMapAsText(MIRPath.PHRS_DIR + "abbr_tok.txt", sm);

		// FileUtils.writeStringCollectionAsText(MIRPath.PHRS_DIR + "abbr_tok.txt",
		// sfs);
	}

}
