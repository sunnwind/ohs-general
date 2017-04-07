package ohs.corpus.dump;

import java.io.FileInputStream;
import java.io.StringReader;
import java.text.DecimalFormat;
import java.util.List;

import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonObject;
import javax.json.JsonReader;
import javax.json.JsonValue;
import javax.json.stream.JsonParser;
import javax.json.stream.JsonParser.Event;
import javax.json.stream.JsonParserFactory;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.ir.medical.general.MIRPath;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class BioAsqDumper extends TextDumper {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		BioAsqDumper dh = new BioAsqDumper(MIRPath.BIOASQ_COL_RAW_DIR + "allMeSH_2017.json", MIRPath.BIOASQ_COL_LINE_DIR);
		dh.dump();

		System.out.println("process ends.");
	}

	public BioAsqDumper(String inDirName, String outDirName) {
		super(inDirName, outDirName);

		this.batch_size = 50000;
	}

	@Override
	public void dump() throws Exception {
		FileUtils.deleteFilesUnder(outPathName);

		TextFileReader reader = new TextFileReader(inPathName);

		List<String> keys = Generics.newArrayList();

		DecimalFormat df = new DecimalFormat("00000");

		List<String> res = Generics.newArrayList();

		while (reader.hasNext()) {
			reader.printProgress();

			String line = reader.next();

			if (line.startsWith("{\"articles") || line.endsWith("]}")) {
				continue;
			}

			JsonReader jr = Json.createReader(new StringReader(line));
			JsonObject jo = jr.readObject();

			if (keys.size() == 0) {
				for (String key : jo.keySet()) {
					keys.add(key);
				}
			}

			String pmid = jo.getString("pmid");
			String title = jo.getString("title");
			String abs = jo.getString("abstractText");
			String year = jo.getString("year", "");
			String journal = jo.getString("journal");

			String mesh = "";

			{
				JsonArray ja = jo.getJsonArray("meshMajor");
				List<String> items = Generics.newArrayList();

				for (JsonValue item : ja) {
					items.add(item.toString());
				}

				mesh = StrUtils.join("|", StrUtils.unwrap(items));
			}

			List<String> items = Generics.newArrayList();
			items.add(pmid);
			items.add(journal);
			items.add(year);
			items.add(mesh);
			items.add(title);
			items.add(abs);

			String out = StrUtils.join("\t", StrUtils.wrap(items));
			res.add(out);

			if (res.size() == batch_size) {
				String outFileName = String.format("%s/%s.txt.gz", outPathName, df.format(batch_cnt.getAndIncrement()));
				FileUtils.writeStringCollectionAsText(outFileName, res);
				res.clear();
			}

			// System.out.println(out);

			// writer.write(out + "\n");

		}
		reader.printProgress();
		reader.close();

		if (res.size() > 0) {
			String outFileName = String.format("%s/%s.txt.gz", outPathName, df.format(batch_cnt.getAndIncrement()));
			FileUtils.writeStringCollectionAsText(outFileName, res);
		}
	}

	/**
	 * http://www.journaldev.com/2315/java-json-example
	 * 
	 * @throws Exception
	 */
	public void test2() throws Exception {

		JsonParserFactory factory = Json.createParserFactory(null);
		JsonParser jp = factory.createParser(new FileInputStream(MIRPath.BIOASQ_COL_RAW_DIR + "allMeSH_2017.json"));

		while (jp.hasNext()) {
			Event event = jp.next();
			switch (event) {
			case KEY_NAME:
				String s = jp.getString();
				break;
			case VALUE_STRING:
				String s2 = jp.getString();
				break;
			case VALUE_NUMBER:
				break;
			case VALUE_FALSE:
				break;
			case VALUE_TRUE:
				break;
			case VALUE_NULL:
				// don't set anything
				break;
			default:
				// we are not looking for other events
			}
		}
	}

}
