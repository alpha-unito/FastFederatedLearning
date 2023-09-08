//
// Created by gmittone on 9/7/23.
//

#ifndef FASTFEDERATEDLEARNING_DATA_STUCTS_HPP
#define FASTFEDERATEDLEARNING_DATA_STUCTS_HPP

struct View {
    torch::Tensor view;
    uint32_t id_square;
    uint32_t id_frame;

    View() {}

    explicit View(const uint32_t &s, const uint32_t &f) : id_square{s}, id_frame{f} {}

    explicit View(const View *t) : id_square{t->id_square}, id_frame{t->id_frame} {}
};

template<class Archive>
void save(Archive &archive, View const &f) {
    std::ostringstream oss;

    oss.write((const char *) &f.id_square, sizeof(f.id_square));
    oss.write((const char *) &f.id_frame, sizeof(f.id_frame));

    std::ostringstream stream;
    torch::save(f.view, stream);
    std::string buffer = stream.str();
    uint32_t size = buffer.length();
    oss << size << buffer;

    archive(oss.str());
}

template<class Archive>
void load(Archive &archive, View &f) {
    std::string oss;
    archive(oss);
    std::istringstream iss(oss);
    uint32_t size;

    iss.read((char *) &f.id_square, sizeof(f.id_square));
    iss.read((char *) &f.id_frame, sizeof(f.id_frame));
    iss.read((char *) &size, sizeof(size));

    char *buffer = new char[size];
    iss.read(buffer, size);

    std::string string_buffer(buffer, size);
    // Creare uno string buffer è necessario perché char* torch load lo interpreta come filename
    // TODO: fare overloading di torch::load?
    std::istringstream stream(string_buffer);
    torch::load(f.view, stream);

    delete buffer;
}


struct Frame {
    cv::Mat frame;
    int32_t id_camera;
    uint32_t id_square;
    uint32_t id_frame;
    uint32_t rows;
    uint32_t cols;
    float max;


    Frame() {}

    explicit Frame(const uint32_t &s, const int32_t &c, const uint32_t &f, const uint32_t &row, const uint32_t &col,
                   float max)
            : id_square{s}, id_camera{c}, id_frame{f}, rows{row}, cols{col}, max{max} {}

    explicit Frame(const uint32_t &s, const int32_t &c, const uint32_t &f)
            : id_square{s}, id_camera{c}, id_frame{f}, rows{0}, cols{0} {}

    explicit Frame(const uint32_t &s, const uint32_t &f)
            : id_square{s}, id_camera{-1}, id_frame{f}, rows{0}, cols{0} {}

    explicit Frame(const Frame *t) : id_square{t->id_square}, id_camera{t->id_camera}, id_frame{t->id_frame},
                                     rows{t->rows}, cols{t->cols} {}
};

template<class Archive>
void save(Archive &archive, Frame const &f) {
    std::ostringstream oss;
    uint32_t size = f.frame.total() * f.frame.elemSize();
    uint32_t channels = f.frame.channels();

    oss.write((const char *) &f.id_square, sizeof(f.id_square));
    oss.write((const char *) &f.id_camera, sizeof(f.id_camera));
    oss.write((const char *) &f.id_frame, sizeof(f.id_frame));
    oss.write((const char *) &f.rows, sizeof(f.rows));
    oss.write((const char *) &f.cols, sizeof(f.cols));
    oss.write((const char *) &channels, sizeof(channels));
    oss.write((const char *) &f.max, sizeof(f.max));
    oss.write((const char *) &size, sizeof(size));
    oss.write((const char *) f.frame.data, size);
    archive(oss.str());
}

template<class Archive>
void load(Archive &archive, Frame &f) {
    std::string oss;
    archive(oss);
    std::istringstream iss(oss);
    uint32_t size;
    uint32_t channels;

    iss.read((char *) &f.id_square, sizeof(f.id_square));
    iss.read((char *) &f.id_camera, sizeof(f.id_camera));
    iss.read((char *) &f.id_frame, sizeof(f.id_frame));
    iss.read((char *) &f.rows, sizeof(f.rows));
    iss.read((char *) &f.cols, sizeof(f.cols));
    iss.read((char *) &channels, sizeof(channels));
    iss.read((char *) &f.max, sizeof(f.max));
    iss.read((char *) &size, sizeof(size));
    f.frame = cv::Mat(f.rows, f.cols, CV_8UC(channels)).clone();
    iss.read((char *) f.frame.data, size);
}

#endif //FASTFEDERATEDLEARNING_DATA_STUCTS_HPP
