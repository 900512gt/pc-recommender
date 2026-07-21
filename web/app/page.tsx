import Link from "next/link";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-360 flex-1 flex-col justify-center px-6 py-32">
      <h1 className="max-w-2xl text-6xl font-semibold tracking-tight">
        用遺傳演算法，找到最適合你的 PC 組合
      </h1>
      <p className="mt-6 max-w-xl text-lg text-text-muted">
        整合 PTT / 巴哈姆特零件口碑分析，在預算內自動搜尋相容且評價最好的組裝方案。
      </p>
      <div className="mt-10">
        <Link
          href="/builder"
          className="inline-block rounded-sm bg-accent px-5 py-2.5 text-sm font-medium text-accent-inverse transition-opacity hover:opacity-90"
        >
          開始建置配置 →
        </Link>
      </div>
    </main>
  );
}
