import Image from "next/image";
import Link from "next/link";
import UserButton from "@/modules/auth/ui/components/user-button";

export default function Navbar() {
  return (
    <nav className="fixed z-50 flex w-full items-center justify-center p-4">
      <div className="flex h-16 w-full md:w-[50%] items-center justify-between rounded-full border-2 p-4">
        <Link
          href="/"
          className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
        >
          <Image src="/assets/advision.svg" alt="logo" width={28} height={28} className="" />
          <span className="text-2xl tracking-tight">
            AdVision
          </span>
        </Link>

        <div className="flex items-center gap-4">
          <UserButton />
        </div>
      </div>
    </nav>
  );
}
