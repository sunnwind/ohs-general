package ohs.java.study;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import ohs.math.ArrayMath;
import ohs.utils.Generics;

public class ExecuteExam {
	static class MyCaller implements Callable<String> {

		public MyCaller(ExecutorService es) {
			threadPoolExecutor = (ThreadPoolExecutor) es;

			int[] a = ArrayMath.random(0, 10, 10);
			
			if(a.length > 0){
				System.out.println();
			}

			id = ArrayMath.sum(a);
		}

		ThreadPoolExecutor threadPoolExecutor;

		@Override
		public String call() throws Exception {
			int poolSize = threadPoolExecutor.getPoolSize();
			String threadName = Thread.currentThread().getName();
			System.out.println("[총 스레드 개수: " + poolSize + "] 작업 스레드 이름: " + threadName);

			try {
				if (Thread.currentThread().getName().contains("thread-1")) {
					Thread.sleep(1);
					System.out.println("slept...");
				}
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			// int value = Integer.parseInt("숫자");
			return "id: " + id + "\t" + threadName;
		}

		private int id;

		public void setId(int id) {
			this.id = id;
		}

	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		test2();

		System.out.println("process ends.");
	}

	public static void test2() throws Exception {
		ExecutorService es = Executors.newFixedThreadPool(2);
		List<Future> fs = Generics.newArrayList();

		for (int i = 0; i < 10; i++) {

			MyCaller c = new MyCaller(es);
			// c.setId(i);

			// executorService.execute(runnable);
			Future<String> f = es.submit(c);
			fs.add(f);
		}

		for (Future<String> f : fs) {
			String name = f.get();

			System.out.printf("called: %s\n", name);
		}

		System.out.println("waiting...");
		es.shutdown();
	}

	public static void test1() throws Exception {
		ExecutorService executorService = Executors.newFixedThreadPool(2);

		for (int i = 0; i < 10; i++) {
			Runnable runnable = new Runnable() {
				@Override
				public void run() {
					ThreadPoolExecutor threadPoolExecutor = (ThreadPoolExecutor) executorService;

					int poolSize = threadPoolExecutor.getPoolSize();
					String threadName = Thread.currentThread().getName();
					System.out.println("[총 스레드 개수: " + poolSize + "] 작업 스레드 이름: " + threadName);

					int value = Integer.parseInt("숫자");
				}
			};

			// executorService.execute(runnable);
			executorService.submit(runnable);

			Thread.sleep(10);
		}
		executorService.shutdown();
	}
}
