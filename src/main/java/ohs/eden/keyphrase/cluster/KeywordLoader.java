package ohs.eden.keyphrase.cluster;

import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.types.common.StrPair;
import ohs.types.generic.Counter;
import ohs.types.generic.SetMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class KeywordLoader {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		KeywordLoader loader = new KeywordLoader();
		loader.open();

		{
			KeywordData kwdData = new KeywordData();
			kwdData.readObject(KPPath.KYP_DATA_SER_FILE.replace("_data", "_data_clusters"));

			Set<Integer> confirmed = Generics.newHashSet();

			String confirmFileName = KPPath.KYP_DIR + "cids_confirmed.ser.gz";

			if (FileUtils.exists(confirmFileName)) {
				for (int cid : FileUtils.readIntegers(KPPath.KYP_DIR + "cids_confirmed.ser.gz")) {
					confirmed.add(cid);
				}
			}

			loader.loadKeywords("Keywords", kwdData, confirmed);
			// loader.loadKeywordsMap("Keywords_Map", kwdData);
		}
		//
		// {
		// String tableName = "Keywords_Map_Other3P";
		// loader.loadKeywordsMap(tableName, KPPath.KYP_DIR + "keyword_other_3p.txt.gz");
		// }
		//
		// {
		// String tableName = "Keywords_Map_JST";
		// loader.loadKeywordsMap(tableName, KPPath.KYP_DIR + "keyword_jst_3p.txt.gz");
		// }
		//
		// {
		// String tableName = "Keywords_Map_Hanlim_KorEng";
		// loader.loadKeywordsMap(tableName, KPPath.KYP_DIR + "keyword_hanlim-kor-eng_3p.txt.gz");
		// }
		//
		// {
		// String tableName = "Keywords_Map_Hanlim_EngKor";
		// loader.loadKeywordsMap(tableName, KPPath.KYP_DIR + "keyword_hanlim-eng-kor_3p.txt.gz");
		// }

		loader.close();
		System.out.println("process ends.");
	}

	private Connection con;

	private int batch_size = 100000;

	public KeywordLoader() {

	}

	public void close() throws Exception {
		con.close();
	}

	public void createKeywordMapTable(String tableName) throws Exception {
		String query = String.format(
				"CREATE TABLE `%s` (`cn` text NOT NULL, `kwdid` int(11) NOT NULL, KEY `CN` (`cn`(45)), KEY `kwdid` (`kwdid`)) ENGINE=MyISAM DEFAULT CHARSET=utf8;",
				tableName);

		Statement stmt = con.createStatement();
		stmt.executeUpdate(query);

		con.commit();
		stmt.close();
	}

	private void deleteTable(String tableName) throws Exception {
		DatabaseMetaData md = con.getMetaData();
		ResultSet rs = md.getTables(null, null, tableName, null);

		if (rs.next()) {
			String query = "delete from " + tableName;
			Statement stmt = con.createStatement();
			int deletedRows = stmt.executeUpdate(query);

			if (deletedRows > 0) {
				System.out.println("Deleted All Rows In The Table Successfully...");
			} else {
				System.out.println("Table already empty.");
			}

			con.commit();
			stmt.close();
		}
	}

	private void loadKeywords(String tableName, KeywordData kwdData, Set<Integer> confirmed) throws Exception {
		// printMetaData(tableName);
		// deleteTable(tableName);

		SetMap<Integer, Integer> clstToKwds = kwdData.getClusterToKeywords();
		Map<Integer, Integer> clstToLabel = kwdData.getClusterToLabel();

		int kwd_cnt = 0;

		/*
		 * kwdid, kor_kwd, eng_kwd, cid, is_label, kwd_freq
		 */
		String sql = String.format("insert into %s values (?,?,?,?,?,?,?,?,?)", tableName);
		PreparedStatement pstmt = con.prepareStatement(sql);

		Timer timer = Timer.newTimer();

		List<Integer> cids = Generics.newArrayList();

		{
			Counter<Integer> c = Generics.newCounter();
			for (int cid : clstToKwds.keySet()) {
				c.setCount(cid, clstToKwds.get(cid).size());
			}

			cids = c.getSortedKeys();
		}

		DecimalFormat df = new DecimalFormat("000000000000000");

		for (int i = 0; i < cids.size(); i++) {
			int cid = cids.get(i);

			Counter<Integer> c = Generics.newCounter();

			String clstID = String.format("ADTPC%s", df.format(i));
			String tmpClstID = String.format("ECT%s", df.format(i));
			int is_confirmed = 0;

			if (confirmed.contains(cid)) {
				is_confirmed = 1;
			}

			for (int kwdid : clstToKwds.get(cid)) {
				int kwd_freq = kwdData.getKeywordFreqs()[kwdid];
				c.setCount(kwdid, kwd_freq);
			}

			for (int kwdid : c.getSortedKeys()) {
				StrPair kwdp = kwdData.getKeywordIndexer().getObject(kwdid);
				String[] two = kwdp.asArray();

				two = StrUtils.unwrap(two);

				String korKwd = two[0];
				String engKwd = two[1];

				// if (korKwd.length() == 0) {
				// continue;
				// }

				int is_label = clstToLabel.get(cid) == kwdid ? 1 : 0;
				int kwd_freq = kwdData.getKeywordFreqs()[kwdid];

				// System.out.println(kwdStr);

				if (++kwd_cnt % batch_size == 0) {
					int[] res = pstmt.executeBatch();
					con.commit();
					System.out.printf("\r[%d, %s]", kwd_cnt, timer.stop());
				}

				pstmt.setInt(1, kwdid);
				pstmt.setString(2, korKwd);
				pstmt.setString(3, engKwd);
				pstmt.setInt(4, cid);
				pstmt.setBoolean(5, is_label > 0 ? true : false);
				pstmt.setInt(6, kwd_freq);

				pstmt.setString(7, clstID);
				pstmt.setString(8, tmpClstID);
				pstmt.setInt(9, is_confirmed);

				pstmt.addBatch();
			}
		}

		pstmt.executeBatch();
		con.commit();
		pstmt.close();

		System.out.printf("\r[%d, %s]\n", kwd_cnt, timer.stop());

	}

	private void loadKeywordsMap(String tableName, KeywordData kwdData) throws Exception {
		printMetaData(tableName);
		deleteTable(tableName);

		Timer timer = Timer.newTimer();

		/*
		 * cn, kwdid
		 */
		String sql = String.format("insert into %s values (?,?)", tableName);

		PreparedStatement pstmt = con.prepareStatement(sql);
		int num_kwds = 0;

		for (int kwdid : kwdData.getKeywordToDocs().keySet()) {
			for (int docid : kwdData.getKeywordToDocs().get(kwdid)) {
				String cn = kwdData.getDocumentIndxer().getObject(docid);

				if (++num_kwds % batch_size == 0) {
					pstmt.executeBatch();
					con.commit();
					System.out.printf("\r[%d, %s]", num_kwds, timer.stop());
				}

				pstmt.setString(1, cn);
				pstmt.setInt(2, kwdid);
				pstmt.addBatch();
			}
		}

		pstmt.executeBatch();
		con.commit();
		pstmt.close();

		System.out.printf("\r[%d, %s]\n", num_kwds, timer.stop());

	}

	public void loadKeywordsMap(String tableName, String fileName) throws Exception {
		deleteTable(tableName);
		createKeywordMapTable(tableName);

		TextFileReader reader = new TextFileReader(fileName);
		reader.setPrintNexts(false);
		reader.setPrintSize(batch_size);

		List<String> lines = Generics.newArrayList();

		while (reader.hasNext()) {
			reader.printProgress();

			String line = reader.next();

			if (lines.size() == batch_size) {
				/*
				 * cn, kwdid
				 */
				String sql = String.format("insert into %s values (?,?)", tableName);

				// Keywords_Map
				PreparedStatement pstmt = con.prepareStatement(sql);

				for (String s : lines) {
					String[] parts = s.split("\t");
					String cn = parts[0];
					int kwdid = Integer.parseInt(parts[1]);

					pstmt.setString(1, cn);
					pstmt.setInt(2, kwdid);
					pstmt.addBatch();
				}

				pstmt.executeBatch();
				con.commit();
				pstmt.close();

				lines = null;
				lines = Generics.newArrayList();
			} else {
				lines.add(line);
			}
		}
		reader.printProgress();
		reader.close();

	}

	public void open() throws Exception {
		Class.forName("com.mysql.jdbc.Driver");
		String[] lines = FileUtils.readFromText(KPPath.DB_ACCOUNT_FILE).split("\t");

		StrUtils.trim(lines);

		String url = String.format("jdbc:mysql://%s:3306/authority", lines[0]);
		String id = lines[1];
		String pwd = lines[2];

		con = DriverManager.getConnection(url, id, pwd);
		con.setAutoCommit(false);
	}

	private void printMetaData(String tableName) throws Exception {
		DatabaseMetaData meta = con.getMetaData();
		ResultSet resultSet = meta.getColumns(null, null, tableName, null);
		while (resultSet.next()) {
			String name = resultSet.getString("COLUMN_NAME");
			String type = resultSet.getString("TYPE_NAME");
			int size = resultSet.getInt("COLUMN_SIZE");

			System.out.println("Column name: [" + name + "]; type: [" + type + "]; size: [" + size + "]");
		}
	}

}
