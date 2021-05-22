CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?Ł$�/      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��H   max       P��`      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �D��   max       >o      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?:�G�{   max       @F
=p��     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33333    max       @vpz�G�     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @M@           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @��          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       >["�      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B0*�      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�>b   max       B0=�      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @#ӏ   max       C�bc      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @��   max       C�h�      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��H   max       PV�      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�X�e+�   max       ?���䎋      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �<j   max       >o      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�
=p��   max       @F
=p��     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @vpz�G�     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @M@           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @�e           �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         DJ   max         DJ      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��+I�   max       ?��|����     �  T      	                           U   (   U                     +      Q         	            '   ,      /      	   .            �   #      >      .   k                  	   o               /   :   )   ?N���N~\4N.�`O7'�N�jOc��N��N�6�N|R!O+�yP��`O���P;w{O!^BNdO��O�xAN^�'N%fCP^�GNQ�PQ��O��N%�JO4�Oe�O͚VO���O��(O���NOl�O��+O��N��O�TOu�N���O�fLP�$*O��N$ݜP<>�OR�P\j�PW�OF3qM��HNg�N��^N ٗNX�1O�KLN<��N�Q�NpT�N�s�O��nOy��NԋpO�T�D���ě��e`B�t��ě�:�o;D��;�o<o<o<t�<49X<T��<e`B<�o<�C�<�t�<�t�<�1<�9X<�j<���<�`B<�h<�h<�h<�<�<��=\)=��=#�
=#�
=,1=49X=8Q�=<j=<j=@�=@�=D��=aG�=aG�=aG�=e`B=q��=q��=}�=�%=�%=�o=�+=�C�=�t�=���=ȴ9=�/=�;d=��>oqiikost���������{tqqwuz����������zwwwwwww{���������{wwwwwwww��������

������'09<>IIIE<20''''''''��������������������GBN[ggptttqg[VPNGGGG`^hjt���������|tlh``1/6BOUSOGB>611111111nnt}�������������utn�����/PUgicH7/��� #/<VagiaaiaUHE<,$ �����BIFE?5)����������� ��������������� ��������������������������������������

������!!#'0<?A<50+#!!!!!!V[gjtzztjgc[VVVVVVVV[\qz��������������t[������������������������OY[TMIB9��������������Y[_chtxwtjh[YYYYYYYYdchhit�����������thd\hty��������������h\�������������������������)-6;975)���<99;;@HOUanz���znUH<3HOPYam�������maTH;3����������������������������

�����#%/06:620#�����������������������)6BKPNJ>6) ����������
#&*)#������))))�������}~���������������������)>N[x���tP5�������������������������

�������������������5A@5)����sqrt��������������tsrrnpz�������������zr���� 
/573/)#
����)*-15675-)
#'#


��������������������((*568BA6*((((((((((����������������������������

�����
#$#
^abmz�����zmba^^^^^^������� ��������������������������������jhainz����������wtqj���������

����UV\antz{|{zrna_XUUUUwqnnsz������������zw�<�H�U�a�n�z�{�~�z�n�f�a�U�T�H�<�<�8�<�<E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͼ���������������z��������������������������������������ߺ޺ݺ��ﻞ�������������������������������������������������������������������������#�#�������������������������������������������������������������ʼּ���ּʼż������ļʼʼʼʼʼʼʼ��Z�g�s���������������s�g�Z�N�A�@�F�N�O�Z�"�G�m���������m�^�;�.�"�
��ھȾʾ߾��"������������������ù÷éèìð�������#�<�U�b�l�r�s�o�]�U�I�
���������������#���ûлٻۻܻԻлû���������������������������������������������ſ�������������ƻ�����'�4�5�=�;�4�'������������r�����������������������}�f�S�Q�Y�b�r�����������������w�r�m�r�~����������������������y�x�y�������������������A�M�s������������������M�4�(����.�AÇÓàâçáàÓÓÇÄ�~ÇÇÇÇÇÇÇÇ���л����� � ���лû������x�^�O�h�����ѿݿ���
�������ѿĿ������������ȿѾM�T�Z�f�o�m�f�Z�W�M�J�J�M�M�M�M�M�M�M�M�l�x�������������������x�l�b�_�Z�V�U�`�l��*�<�H�Q�U�]�U�P�H�<�/�*�#�������������������������������s�g�\�\�`�g�s�����������	����	�����������������������a�m�z�������������������������v�m�d�a�aĚĲĿ��������������ĿĦĚēěęęĐđĚ�
�������
�� ���
�
�
�
�
�
�
�
������������������������������������������������!�)�!��������߼ؼ޼��ÓàâìîìàÓÐÉÓÓÓÓÓÓÓÓÓÓ�ɺֺ޺ݺֺͺѺʺ��������~�s�s�~�����������	��"�/�7�@�G�H�@�;�/�"���	���������Z�e�b�f�g�f�b�Z�Z�M�M�F�I�K�M�P�Z�Z�Z�Z�y���������������½�������������y�s�p�y��)�6�B�I�G�O�c�m�[�6�)�"�������������)�6�8�=�B�G�B�6�1�)���������!�)�������������������������������������¿���������� �-�'�
����������µ¥«½¿��������(�(��������ܿܿݿ޿�����������������������������V�A�8�6�:�g���)�B�[�c�t���h�[�N�)�������������)ƚƧƳ����������������ƶƳƧƚƎƇƇƕƚ¦¨¬¦¢¦¦¦¦¦¦¦¦�ʾ������������������������þʾ˾ʾʾʾ��m�z�������������������z�y�m�e�h�m�m�m�m�S�`�d�l�m�l�`�S�M�R�S�S�S�S�S�S�S�S�S�S����������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DyDsDpD{D�ǔǡǪǦǡǗǔǑǈǆǈǎǔǔǔǔǔǔǔǔ����������������ŹŷůŲŹż��������������*�6�6�C�G�K�C�6�5�*����������l�x�����������������x�l�c�k�l�l�l�l�l�l�ɺֺ�����ߺֺɺ�������������������EiEuE�E�E�E�E�E�E�E�E�E�E�E�EuEiEhEiEfEi�Ļлܻ��ܻһлû��������������ĻĻĻļM�Y�f�r��������x�f�Y�M�@�4�*�'�+�4�G�M D B [ ; [ ; H h ( : & /   e ] 6 N Y 6 < P ? _  E , , / J q W & T Q P q ; T ; A % 0 4 2 A u f G 4 2 6 ? . l H < 4 $ /    �  m  �  N  �  �  #  �  ~  [  r  +  L  �  k  *  k  [  �  l  �  7  a  v  �  �  !  m    �  �  -  2        8  �  a  P    �  �  |  �  F  �    $  Y  j  Q  �  �  �  |  �  �  S����o�t�<o�o<ě�<T��<u<�1<��
=�v�=P�`=ȴ9=o<�9X<�=t�<�j<���=�o<�=��=T��=o=��=H�9=aG�=0 �=�O�=���=0 �=� �=ix�=L��=�Q�=��=Y�=�7L>["�=��=e`B==��=���>&�y=�\)=��=�C�=��=�O�=��>49X=��=�1=� �=�S�>�->)��>'�>C��B	�B��B)O�B#.:B&
�B�'B�B�B7vB,B�zBBއB#PB�>B"O�B"��B%��B	@�B�_B �fB��Bz�B�UB�wB�B1(BBfA�!�B`�Bo�B%2DB! xB�Bs�B��B*!�B��B�RB�&B!B
�B�B��BP%B8�B$��B2�B0*�Bu_B�B!�A��B��B,��B�B(kB��BI�B
/AB=�B)�UB"ԙB&5iBÞB��B�
B@B��B¡B�B��B#>_B�MB"8B"@BB%��B	XxBȼB ��B�BJ/BB@B��B'�B>uB�A��B@TB�B%?�B!>8B7nB��BaB)ŝB��B�BH B��B3�B٦B��B=MB;sB$�wB>�B0=�BDBB�B)�A�>bBCsB,��B@bBS$BIB�QA�1DC�bc@�x2@S��@�a�A�~A�A@���A 8�A��Ab�]AЩ=A�L@���A���@¯�@��@�4Ao��AA�nAʆ�@�A~�5A?)�@��lA�A���A�<:A��A⻲A��A�HA8\A�.
@#ӏA�bA=��A �A�b�A�U[A�[A�1A�WnA�)A�;�B:�A���AL�A�iAA61AuC���Bl�A�]	A�߇@�L�@.�C�	)@�V�@�M�A�z�C�h�@�F�@M�@��uAЃA��^@���A �EA���Ag �Aѓ'A�}�@�A���@��G@��@� JAo�AB��A���@�ՋAA?ڦ@�f6A�A�y�A��(A�3AℰA��A��
A�A��@��A��'A?�A �\A��Aր�A�yAA���A���A��jA�d�B�0A�z�AM�A�zpA��At�=C��BK	A�|�A�s=@���@+P�C���@�n@��A      	                           U   )   V                     +      Q         
            '   -      /      	   /            �   $   	   ?      .   l                  
   o               /   :   )   @                                 9   !   )                     1      5                        !               #            ;         /      1   '                                                                              +      !                           )                                                            '      /                                                N��N~\4N.�`O��N�jN�{`N��N�6�N]�bN�P'�O|��O�"�N۸NdO��OZ�N^�'N%fCO�
NQ�P��O٬N%�JO4�Oe�O���Om�HO��POb�NOl�O���N��pN��O�B�Ou�N���N��IO���N6HzN$ݜPːOR�PV�O��OF3qM��HNg�N�_N ٗNX�1O
:�N<��N�Q�NpT�N�s�O��nOR��NԋpO�T  �    �    �  u  A  �  �  �    #  	J  %  ;  8  T  '  [  �    �  �  �  �  s  �  �  f  �  �  -  /  �  �  6  �  q  �    �  b  g  �  �    �  �  �  �  �  �  �  !  �    �  �  �  l�<j�ě��e`B�ě��ě�<49X;D��;�o<t�<D��=C�<�C�=C�<���<�o<�C�<��
<�t�<�1='�<�j=8Q�=\)<�h<�h<�h=\)<��=+=@�=��=49X='�=,1=H�9=8Q�=<j=]/=�F=�%=D��=}�=aG�=e`B=���=q��=q��=}�=��=�%=�o=�G�=�C�=�t�=���=ȴ9=�/=�=��>oljlptx���������tllllwuz����������zwwwwwww{���������{wwwwwwww��������

������'09<>IIIE<20''''''''��������������������GBN[ggptttqg[VPNGGGG`^hjt���������|tlh``206BOTQOEB@622222222srtx���������|tssss����
/:JWWN</#���"%(/<PU^b`\YUH<:/($"�����18>?=5)��������������������������� ��������������������������������������
��������!!#'0<?A<50+#!!!!!!V[gjtzztjgc[VVVVVVVV��������������������������������������������)BGMMIB9*�����������	�����Y[_chtxwtjh[YYYYYYYYdchhit�����������thd\hty��������������h\�������������������������),55:85)�;;<<AHQUanz���znUH<;XWYY]agmz��������zaX���������������������������������"#$-059610#�����������������������)6BHKKB;6)$��������
#&*)#������))))�������������������������� ���)BMRRKC5) �����������������������

������������������.5<=5)����sqrt��������������tsssoq{�������������{s������
#)--*$
���)*-15675-)
#'#


��������������������((*568BA6*((((((((((���������������������������


�������
#$#
^abmz�����zmba^^^^^^������� ��������������������������������jhainz����������wtqj��������

����UV\antz{|{zrna_XUUUUwqnnsz������������zw�H�U�a�n�y�z�}�z�n�a�a�a�U�H�@�>�H�H�H�HE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͼ���������������z��������������������������������������������ﻞ���������������������������������������������������������������������������������#�#�������������������������������������������������������������ʼּ���ּʼǼ������żʼʼʼʼʼʼʼ��Z�g�k�s���������s�g�Z�N�K�K�N�W�Z�Z�Z�Z�"�;�T�m�������y�`�;�.�"���������"������
��� ��������ùôñöù���������#�<�I�U�_�e�f�b�U�I�0�#�
����������
�#�ûϻлԻӻл̻û��������������������û�������������������������ſ�������������ƻ�����'�4�5�=�;�4�'������������r���������������������u�r�f�[�V�Y�e�r�����������������w�r�m�r�~����������������������y�x�y�������������������M�f�s���������������s�f�Z�M�C�<�B�@�MÇÓàâçáàÓÓÇÄ�~ÇÇÇÇÇÇÇÇ�����лܻ���������ܻлû����x�g�x���ѿݿ������
�	�������ݿѿѿɿɿѿѿѾM�T�Z�f�o�m�f�Z�W�M�J�J�M�M�M�M�M�M�M�M�l�x�������������������x�l�b�_�Z�V�U�`�l��*�<�H�Q�U�]�U�P�H�<�/�*�#���������������������������������g�e�b�g�s�t�����������	�����	���������������������m�z�������������������������x�n�f�c�k�mĦĳĿ��������������������ĿĳĦğĠģĦ�
�������
�� ���
�
�
�
�
�
�
�
���������������������������������������˼�������!�#�!���������ټ���ÓàâìîìàÓÐÉÓÓÓÓÓÓÓÓÓÓ���ɺ׺Ժʺκƺ��������~�w�x�~�������������	��"�/�7�@�G�H�@�;�/�"���	���������Z�e�b�f�g�f�b�Z�Z�M�M�F�I�K�M�P�Z�Z�Z�Z�������������������������������������������)�-�4�6�5�1�)������������������)�5�6�:�6�,�)�����&�)�)�)�)�)�)�)�)����������������������������������������������
��(�#�
�����������¼®®·�˿�������(�(��������ܿܿݿ޿�����������������������������X�C�:�8�M�g�����)�B�V�c�k�k�d�[�N�B�)����������ƚƧƳ����������������ƶƳƧƚƎƇƇƕƚ¦¨¬¦¢¦¦¦¦¦¦¦¦�ʾ������������������������þʾ˾ʾʾʾ��m�z�������������������~�z�m�j�k�m�m�m�m�S�`�d�l�m�l�`�S�M�R�S�S�S�S�S�S�S�S�S�S����������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ǔǡǪǦǡǗǔǑǈǆǈǎǔǔǔǔǔǔǔǔ����������������ŹŷůŲŹż��������������*�6�6�C�G�K�C�6�5�*����������l�x�����������������x�l�c�k�l�l�l�l�l�l�ɺֺ�����ߺֺɺ�������������������EuE�E�E�E�E�E�E�E�E�E�E�E�E�EuEkEjElEmEu�Ļлܻ��ܻһлû��������������ĻĻĻļM�Y�f�r��������x�f�Y�M�@�4�*�'�+�4�G�M C B [ 4 [ 5 H h * - + ,   & e ] 5 N Y  < M  _  E ) & + $ q N % T L P q = " K A & 0 0 - A u f K 4 2 ! ? . l H < + $ /  �  �  m  K  N  �  �  #  n  �  �  �    �  �  k  �  k  [  Q  l  �  L  a  v  �  )  �  E  �  �  h    2  �      +  �  U  P  �  �  �  �  �  F  �  �  $  Y  &  Q  �  �  �  |  �  �  S  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  DJ  �  �  �  �  �  �  �  �  q  T  6    �  �  �  �  d  A    �          �  �  �  �  �  �  �  �  �  �          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �           �  �  �  �  �  h  :    �  �  �  �  �  g  �  �  �  �  �  �  �  �  �  �  �  {  u  o  i  c  _  \  Z  W  T  �  �  �    !  <  U  i  s  s  m  [  C  "  �  �  o    �  j  A  9  1  '        �  �  �  �  �  y  X  1    �  �  �  \  �  �  �  �  �  �  �  ~  g  W  R  N  J  "  �  �  �  R    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
  u  |  �  �  ~  |  �  z  n  _  O  >  *      �  �  C  �  �  �  Z  �  �  �  �    �  �  �  h    �  2  �  X  �  .  &  s  �      "      �  �  �  |  r  �  �  �  �  j  O    �  �    �  �  	+  	D  	J  	@  	)  	  �  �  �  3  �  g  �        v        !  $  %       �  �  �  �  `  .  �  �  �  }  �  ^  ;  7  2  .    �  �  �  �  �  �  �  t  ^  B  &          8  7  2  *       	  �  �  �  �  �  �  �  �  �  �  �  /  y  2  H  S  R  M  E  >  =  >  :  .    �  �  �  t  D    �  h  '                �  �    	        �  �  �  |  ;   �  [  T  M  E  >  4  $      �  �  �  �  �  �  }  R  (   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  X  �  �    *      #  *  &  "        �  �  �  �  �  v  Z  =    �  �  �    m  �  �  �  �  �  �  ]  c  H    �    s  �  �  �  �  a  �  �  �  �  �  �  �  �  �  t  N    �  �  X    �    \  �  �  �  �  �  �  �  �  �  �  �  �  u  f  V  G  7  '      �  �  �  �  s  a  N  9  #    �  �  �  �  �  |  |  }  �  �  s  D    �  �  �  �  �  �  �  �  �  �  }  d  E  #    �  R  �  �  �  �  �  �  �  �  �  �  �  �  }  Y  /  �  �  `    w  �  �  �  �  �  �  �  �  �  �    _  <    �  �  �  �  �  b  K  f  ]  N  >  +      �  �  �  �  }  D  �  �  >  �  c  �  <  G  E  �  �  �  �  �  �  �  �  {  b  F    �    	  �  �  �  �  �  �  x  Y  ;       �  �  �  �  �  �  k  q  }  �  �  �  "  +  %    �  �  �  [    �  }  0  �  �    y  �  �  �     .  .  &        �  �  �  �  }  Q    �  �  Q    �  }  �  �  �  �  �  �  �  �  q  E    �  �  �  N    �  �  m  2  �  �  �  �  �  �  �  v  L  (    �  �  �  x  4  �  p  �  �  6      �  �  �  �  �  �  ]  3    �  �  u  3  �  �  %   �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  X  C  .  %  !    N  K  J  I  G  N  R  h  h  `  W  H  -  
  �  �  b    e   �  5  �    �  �  �  �  t  �  �  ~  L  �  z  �  �  �  
�    �  B  B  R  l  �  �  �  �  �    �  �  �  �  6  �  �  @  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  =    A  S  b  [  >         5  M  D  )  
  �  �    ~  �  �    g  Q  <  &    �  �  �  �  �  �  o  S  4      �  �  �  �  �  �  �  �  a  7  
  �  �  u  <  �  �  X  �  �  /  �  �  (  �    U  y  �  �  �  r  ]  @    �  9  
�  
  	a  �  %  �  ?          �  �  �  �  �  �  r  [  B  $  �  �    =   �   �  �  �  �  u  _  I  3    �  �  �  |  U  .    �  �  �  d  ;  �  �  �  |  o  a  T  H  =  2  (        �  �  �  q  V  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  %  �  �  �  �  �  �  �  y  a  D  &    �  �  �  �  [  0     �   �   n  �  �  �  �  z  f  Q  ;  $    �  �  �  �  \  "  �  �  x  <  �  �  �    A  �  �  �  �  �  �  u    r  �  �  �  q  
  q  �  �  �  �  �  �  �  �  �  r  b  Q  =  '    �  �  �  q  B  !  �  �  �  �    h  N  3    �  �  �  l  ;  �  �  y  M  3  �    h  S  C  3  %    
  �  �  �  �  �  �     I  d  l  u      �  �  �  �  �  w  P  )  �  �  �  ]  $  �  �  r  8   �  �  �  �  �  N    �  �  U  �  �  b    �  q    �  �  }    N  v  �    p  N  '  �  �  k    
�  
2  	�  	  l  �  +  �  �  �  v  1  �  �  Z    
�  
N  	�  	�  	&  �  W  �  |  �  ;  {  �  l  *  
�  
�  
�  
L  
  	�  	�  	Y  	  �  I  �  =  �  �  "  �  �