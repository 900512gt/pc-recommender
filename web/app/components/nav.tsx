"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "首頁" },
  { href: "/builder", label: "配置建置" },
  { href: "/parts", label: "零件口碑" },
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
