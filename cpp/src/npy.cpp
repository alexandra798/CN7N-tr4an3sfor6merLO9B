#include "npy.hpp"
#include <fstream>
#include <sstream>
#include <cstring>
#include <cctype>
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

static bool starts_with(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && std::memcmp(s.data(), p.data(), p.size()) == 0;
}

static bool read_exact(std::ifstream& f, void* buf, size_t n) {
  f.read(reinterpret_cast<char*>(buf), static_cast<std::streamsize>(n));
  return f.good();
}

static bool parse_header(const std::string& h, std::vector<int64_t>& shape, bool& fortran, std::string& descr) {
  // extremely small parser for {'descr': '<f4', 'fortran_order': False, 'shape': (N,T,L,C), }
  auto find_str = [&](const std::string& key) -> size_t {
    return h.find(key);
  };
  size_t dpos = find_str("'descr'");
  //if (dpos == std::string::npos) dpos = find_str(""descr"");
  if (dpos == std::string::npos) dpos = find_str("\"descr\"");
  if (dpos == std::string::npos) return false;
  size_t colon = h.find(':', dpos);
  //size_t q1 = h.find_first_of("'"", colon);
  size_t q1 = h.find_first_of("'\"", colon);
  //size_t q2 = h.find_first_of("'"", q1 + 1);
  size_t q2 = h.find_first_of("'\"", q1 + 1);
  descr = h.substr(q1 + 1, q2 - q1 - 1);

  size_t fpos = find_str("fortran_order");
  if (fpos == std::string::npos) return false;
  colon = h.find(':', fpos);
  size_t val = h.find_first_not_of(" 	", colon + 1);
  if (val == std::string::npos) return false;
  fortran = starts_with(h.substr(val), "True");

  size_t spos = find_str("shape");
  if (spos == std::string::npos) return false;
  colon = h.find(':', spos);
  size_t p1 = h.find('(', colon);
  size_t p2 = h.find(')', p1);
  if (p1 == std::string::npos || p2 == std::string::npos) return false;
  std::string body = h.substr(p1 + 1, p2 - p1 - 1);

  shape.clear();
  std::stringstream ss(body);
  while (ss.good()) {
    std::string tok;
    if (!std::getline(ss, tok, ',')) break;
    // trim
    size_t a = tok.find_first_not_of(" 	");
    size_t b = tok.find_last_not_of(" 	");
    if (a == std::string::npos) continue;
    tok = tok.substr(a, b - a + 1);
    if (tok.empty()) continue;
    shape.push_back(std::stoll(tok));
  }
  return !shape.empty();
}

bool npy_load_f32(const std::string& path, NpyArray& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) return false;

  char magic[6];
  if (!read_exact(f, magic, 6)) return false;
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;

  uint8_t major = 0, minor = 0;
  if (!read_exact(f, &major, 1) || !read_exact(f, &minor, 1)) return false;

  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t hl16 = 0;
    if (!read_exact(f, &hl16, 2)) return false;
    header_len = hl16;
  } else if (major == 2) {
    uint32_t hl32 = 0;
    if (!read_exact(f, &hl32, 4)) return false;
    header_len = hl32;
  } else {
    return false;
  }

  std::string header(header_len, '\0');
  if (header_len > 0) {
    if (!read_exact(f, &header[0], header_len)) return false;  // ✅ 可写
  }


  bool fortran = false;
  std::string descr;
  std::vector<int64_t> shape;
  if (!parse_header(header, shape, fortran, descr)) return false;
  if (fortran) return false;
  if (descr != "<f4" && descr != "|f4") return false;

  size_t n = 1;
  for (auto d : shape) n *= static_cast<size_t>(d);
  out.shape = std::move(shape);
  out.data.resize(n);

  if (!read_exact(f, out.data.data(), n * sizeof(float))) return false;
  return true;
}

static std::string make_header_v10(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (shape.size() == 1) oss << ","; // singletons require trailing comma
    if (i + 1 < shape.size()) oss << ", ";
  }
  oss << "), }";
  std::string h = oss.str();
  // pad with spaces and newline so that magic+ver+hlen+header is aligned to 16
  size_t pre = 6 + 2 + 2; // magic + version + header_len(2)
  size_t total = pre + h.size() + 1; // + newline
  size_t pad = 16 - (total % 16);
  if (pad == 16) pad = 0;
  h.append(pad, ' ');
  h.push_back('\n');
  return h;
}

bool npy_save_f32(const std::string& path, const std::vector<int64_t>& shape, const float* data, size_t n) {
  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) return false;

  const char magic[] = "\x93NUMPY";
  f.write(magic, 6);
  uint8_t major = 1, minor = 0;
  f.write(reinterpret_cast<const char*>(&major), 1);
  f.write(reinterpret_cast<const char*>(&minor), 1);

  std::string header = make_header_v10(shape);
  uint16_t hlen = static_cast<uint16_t>(header.size());
  f.write(reinterpret_cast<const char*>(&hlen), 2);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));

  f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(float)));
  return f.good();
}
