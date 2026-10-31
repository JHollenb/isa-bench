// Full-state ptrace tracer for M4: single-steps a static ELF on the real x86-64 CPU and writes, before every
// instruction, RIP, the sixteen GPRs and EFLAGS as 18 little-endian u64 to <trace-file>. At step 0 it also dumps
// every readable mapping of the process (/proc/pid/maps + /proc/pid/mem) to <mem-file> as records
//   u64 start, u64 length, u8 perms[4], bytes[length]
// so the learned machine can start from the exact initial memory image (stack, argv/env, ELF segments).
// Usage: trace_full <elf> <stdout-file> <stderr-file> <trace-file> <mem-file> <max-steps>
#define _GNU_SOURCE
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/personality.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

static void dump_memory(pid_t pid, const char *path) {
    char maps[64], memp[64], line[512];
    snprintf(maps, sizeof maps, "/proc/%d/maps", pid);
    snprintf(memp, sizeof memp, "/proc/%d/mem", pid);
    FILE *m = fopen(maps, "r");
    int fd = open(memp, O_RDONLY);
    FILE *out = fopen(path, "wb");
    if (!m || fd < 0 || !out) { perror("dump"); exit(20); }
    while (fgets(line, sizeof line, m)) {
        unsigned long lo, hi; char perms[5] = {0};
        if (sscanf(line, "%lx-%lx %4s", &lo, &hi, perms) != 3) continue;
        if (perms[0] != 'r' || strstr(line, "[vvar]") || strstr(line, "[vsyscall]")) continue;
        uint64_t len = hi - lo;
        char *buf = malloc(len);
        ssize_t got = pread(fd, buf, len, (off_t)lo);
        if (got != (ssize_t)len) { free(buf); continue; }
        uint64_t start = lo;
        fwrite(&start, 8, 1, out); fwrite(&len, 8, 1, out); fwrite(perms, 1, 4, out); fwrite(buf, 1, len, out);
        free(buf);
    }
    fclose(out); fclose(m); close(fd);
}

int main(int argc, char **argv) {
    if (argc != 7) return 2;
    long max_steps = strtol(argv[6], 0, 10);
    pid_t child = fork();
    if (child < 0) { perror("fork"); return 3; }
    if (!child) {
        int out = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0600);
        int err = open(argv[3], O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (out < 0 || err < 0 || dup2(out, 1) < 0 || dup2(err, 2) < 0) _exit(125);
        close(out); close(err);
        personality(ADDR_NO_RANDOMIZE);
        if (ptrace(PTRACE_TRACEME, 0, 0, 0) < 0) _exit(126);
        execl(argv[1], argv[1], (char *)0);
        _exit(127);
    }
    FILE *tr = fopen(argv[4], "wb");
    int status = 0, result = 0, finished = 0;
    long i = 0;
    if (waitpid(child, &status, 0) < 0) { perror("waitpid"); result = 4; goto done; }
    for (i = 0; i < max_steps; ++i) {
        if (!WIFSTOPPED(status) || WSTOPSIG(status) != SIGTRAP) { result = 5; break; }
        struct user_regs_struct r;
        if (ptrace(PTRACE_GETREGS, child, 0, &r) < 0) { perror("getregs"); result = 7; break; }
        if (i == 0) dump_memory(child, argv[5]);
        uint64_t rec[18] = {r.rip, r.rax, r.rcx, r.rdx, r.rbx, r.rsp, r.rbp, r.rsi, r.rdi,
                            r.r8, r.r9, r.r10, r.r11, r.r12, r.r13, r.r14, r.r15, r.eflags};
        fwrite(rec, 8, 18, tr);
        if (ptrace(PTRACE_SINGLESTEP, child, 0, 0) < 0) { perror("singlestep"); result = 10; break; }
        if (waitpid(child, &status, 0) < 0) { perror("waitpid"); result = 11; break; }
        if (WIFEXITED(status)) { printf("{\"kind\":\"exit\",\"exit_code\":%d,\"steps\":%ld}\n", WEXITSTATUS(status), i + 1); finished = 1; break; }
        if (WIFSIGNALED(status)) { printf("{\"kind\":\"signal\",\"signal\":%d,\"steps\":%ld}\n", WTERMSIG(status), i + 1); finished = 1; result = 12; break; }
    }
done:
    fclose(tr);
    if (!finished) { printf("{\"kind\":\"cap\",\"steps\":%ld}\n", i); kill(child, SIGKILL); waitpid(child, &status, 0); }
    return result;
}
