"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

// 「零件口碑」列表用的是整則評論的正負評標籤與舊的五大維度分數，與配置建置器的面向口碑
// 不是同一套資料，先從導覽列拿掉；零件詳情頁仍可從配置建置器的「看原始評論」與聊天室進入。
const LINKS = [
  { href: "/", label: "首頁" },
  { href: "/builder", label: "配置建置" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <header className="border-b border-border">
      <div className="mx-auto flex h-14 max-w-360 items-center justify-between px-6">
        <Link href="/" className="text-lg font-semibold tracking-tight">
          GA PC Builder
        </Link>
        <nav className="flex items-center gap-6">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`text-sm transition-colors hover:text-text ${
                  active ? "text-text font-medium" : "text-text-muted"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
