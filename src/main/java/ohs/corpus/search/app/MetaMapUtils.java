package ohs.corpus.search.app;

import java.util.Map;

import ohs.io.FileUtils;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class MetaMapUtils {

	public static Map<String, String> readSemanticTypes(String fileName) throws Exception {
		Map<String, String> ret = Generics.newHashMap();
		for (String line : FileUtils.readLinesFromText(fileName)) {
			String[] parts = line.split("\\|");
			String shortName = parts[0];
			String longName = parts[2];
			ret.put(shortName, longName);
		}
		return ret;
	}

	public static Map<String, String> readSemanticGroups(String fileName) throws Exception {
		Map<String, String> ret = Generics.newHashMap();
		for (String line : FileUtils.readLinesFromText(fileName)) {
			String[] parts = line.split("\\|");
			String groupPath = StrUtils.join("-", parts, 0, parts.length - 1);
			String longName = parts[3];
			ret.put(longName, groupPath);
		}
		return ret;
	}
}
