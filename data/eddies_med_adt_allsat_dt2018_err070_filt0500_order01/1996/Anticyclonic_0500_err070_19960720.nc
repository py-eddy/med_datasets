CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��hr�!      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��_   max       Po�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��7L   max       =ȴ9      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F�z�H     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?׮z�H    max       @vy��R     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @N�           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @��`          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �y�#   max       >"��      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�
�   max       B.]�      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�T�   max       B.T�      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?v�   max       C��M      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?g    max       C��m      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          q      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��_   max       P'3�      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���m\��   max       ?�4֡a��      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��7L   max       =ȴ9      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @F�z�H     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
>    max       @vy��R     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @N�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�9`          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Bj   max         Bj      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��	� �   max       ?�4֡a��     �  T         #               $      1         -                        #               &   .            4      $         	      =            q      5   \         l   3      
      
                     N�K�N�/�ON�^N�0O0|N��O��Oe �N�P7H�O�ֵN�|�O�6O�O���N�OD�N�.kO���NgSP aGO�O�"�O�fN��O�7�P)	0O���P0�bN��?PE�IN�x�O��$O*�7M��_N��CNft�P!��N�mN��N�+�Po�N���P��P/OF�Nw�vO�aBO�|�O9CEN��ON�N� bN?J�O+��OA,NX�&N�mN�j8NR�9��7L�D����`B��o�o;D��;D��;�o;��
;��
;ě�;ě�;�`B;�`B;�`B<t�<#�
<#�
<#�
<e`B<�C�<�t�<�t�<ě�<ě�<�/<�`B<�`B<�h<�h<�<�=o=o=o=C�=C�=\)=�w=49X=49X=8Q�=8Q�=8Q�=<j=<j=P�`=T��=ix�=ix�=ix�=m�h=q��=�%=��=��=���=�j=��=ȴ9����������������������������������������eegiot����������tsje��������������������22345BGN[\gmlg\NBB52`]__grt�����tg``````

"0<IMOOKG<30#
+(%$-<HRU^amjaUH<4/+��������������������������
#,AB<0#
�����"/9>IQPD@;/"	���������������������{|���������������gghlrt�����������tgg���
#0IU__WI<0#
��������������������������� ! ��!)6:@BCB6)% kkqu��������������tk366BOZYOEB@633333333������������������z������������������z��������������������bUPIB<0$# #(0<IUXbb��������������������YY\at~����������th_Y�����)6064;<9/)�����������������������)5N[g���{gN�������������������������)BJDC>A@5)�:<BBJNV[agjig_[PNB::)5NX`dfbWB5���� 
#'%##"
��cfhnt���ythhcccccccc����������������������������������������������$&����������


���������-),-/4<>HHOOOLH<:/--���������������������������6OUB)�����)6;BFIIEB>6-)!JK]clt���������mh[OJA;<BN[����������^NJA:8429<AHPT_cgdaUHG@:���������������������������
!##
����������-31)
����������!#$ ��).57:85)!#/368/)'&#��������������������,/<@HIKH=<;50/,,,,,,"/;DHJLKHC;1/#"������*63+�������������������������#
�������
#$%###���������������������

�����������������	�
��
����������������������������¦²¿����������¿²ª¦¦¦¦¦¦¦¦�zÇÏÓàæìêàÓÇ�z�n�d�c�`�a�a�n�z�[�_�c�g�j�g�g�g�[�N�M�N�Q�W�[�[�[�[�[�[�����������������������������������������B�N�[�g�i�g�g�d�[�N�B�>�6�9�B�B�B�B�B�B�S�_�x�������������x�l�S�F�8�-�"�+�9�F�S��������$�&���������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E;�4�=�;�N�y�������s�A��������������"�/�;�H�a�f�h�a�T�H�;�/��	��������"����*�6�7�C�E�C�6�*����������čĦĳĿ������������ĳĦā�t�o�u�x�~Āč�;�@�G�T�`�i�j�`�`�T�G�;�0�.�-�.�3�4�:�;�����������������������r�^�Y�Q�N�Y�r�{�������	������������������"�.�;�G�T�\�T�L�G�;�4�.�"�	�����	�	��"����������׾̾ʾɾ��ľʾ׾����������4�8�(�&������ݽн����Ľ̽���ʼռּ��ּʼü������żʼʼʼʼʼʼʼʿĿѿؿٿ���2�@�(����ѿĿ����������ľZ�l�s�w�v�s�Z�M�A�4�(��%�(�/�4�A�M�Q�Z�(�N�Z�g���������������g�N�A�(�����(����ֺܺκֺ̺׺��������������ﺤ���������������������������������������s���������ʾϾʾ¾�������s�c�V�T�Y�f�s�����"�;�H�a�n�|�z�a�T�/������������������	��"�-�2�5�5�;�:�.��	����������	�"�;�L�L�A�/��	���	���������������	�/�;�H�O�T�V�T�S�J�H�E�;�/�,�,�(�/�/�/�/�uƎƳ�������"�&������Ǝ�u�h�U�U�h�u�I�U�b�b�n�u�{�~�{�t�n�b�_�U�Q�I�G�C�I�I�Ŀѿݿ������������ݿѿ��������Ŀm�y�������������y�m�`�Y�T�G�A�G�O�T�`�m�3�@�C�L�U�S�L�J�@�:�3�3�3�3�3�3�3�3�3�3�N�Z�e�g�q�g�b�Z�N�A�<�5�2�5�A�L�N�N�N�N���������������������������|��������������'�3�;�C�D�3�'�������۹չչ���������	��������������������	��"�-�/�1�/�#�"��	�����������������:�F�_�l�x���������������x�_�S�F�C�:�7�:�@�M�h�g�Y�O�8�'��	���ƻû��лܻ���@��������������ݺ������ ���'�@�M�r������r�Y�@������������B�h�t�x�{�{�r�h�[�B�6�)���%�%���)�B���������������������������s�k�d�n�s�����/�<�H�Q�J�H�<�5�/�+�#� �#�#�/�/�/�/�/�/D�D�D�D�D�D�D�D�D�D�D�D�D�D�DsDqDxD�D�D�¿����������������²�|¦²º¿�
��#�0�<�?�B�A�<�6�0�#��
��������	�
���������������������������������������ؾʾ׾��	����	����˾��������������ʽнݽ��������������ݽֽнϽн�F$F*F(F$FFFFE�E�E�FFFF$F$F$F$F$F$�����
�����
�����������������������`�l�t�y�{�y�|�~�}�y�l�`�S�L�L�M�Q�S�_�`EuE~E�E�E�E�E�E�E|EuEjEiEuEuEuEuEuEuEuEu�a�\�b�d�j�n�o�{ŇňňŇņŅ�~�{�n�a�a�a�zÆÇÓ×ÞßØÓÇ�z�w�n�k�i�n�o�y�z�zǮǯǭǧǡǔǐǌǔǔǡǮǮǮǮǮǮǮǮǮ H 7   n , ! ' ! F H 1 u & 5 9 R I \ D F ` t l J C 8 ] D Y 1 i X + 6 z R v ; ? F n v b E W \ H  9  O i u V : k k f 7 7  �  �  �  �  ~  �  h  �  +  g  �  �  �  >  �  P  �  �    �  8  �  �  M  �  z  �  �  E  �    �  �  ~  Z  �  �  �  4    o  �  A  �  �  �  t  �  E  �  �  (    t  t  �  �  '  &  m�y�#���
<�/�o<49X<49X<ě�=�w<T��=]/=o<49X=P�`<�C�=+<D��<��
<T��<��<�9X=P�`<�j=8Q�=#�
=��=�o=��=]/=u=,1=��T='�=�7L=q��=t�=,1=t�=\=,1=P�`=q��> Ĝ=q��=Ƨ�>O�=e`B=}�>"��=�"�=��w=�7L=��P=�O�=���=\=�v�=�Q�=�"�=��m=�G�B��B ?~B
HBcB5uB	��B%�nB��BJ�B$�A�
�B��B�]B
LPB%��B BЃB�B!�B]�B[�B]�B�B&RkB!��Bw�B_OB �eB�zBR�B�MB^�B^�B��B'2B�sB�B.$B$A�B��B+8�B�%Bq<B��B	��B}4B��B}Ba�Br�B�9B�B��B�A���B.]�BH�B��B!P�B�_B�B 3B
@�Bz�B��B	�B%�\B��B@ B$��A�T�B:B�B
98B%͛B H�B�uBv�BGDBF�B�B
��B(1B&<�B!ʧB��B��B ��B�@B@JB?�B��B��B��BG	B��B�B�B$F�B�zB+BtB�B��B�|B	@TB��B�B4�BH�BE�B��B��B<�B��A��B.T�B:�B>�B!�B��A���A�OLA�<�A��/A� A���@���A�C�moA9�+A���A��dA�JAeQ@�4�@U�Aas.AS$�A/�;A �A��A=�A���@I�2@�HAF��A�!�A\:�A�r�A��B�gA�NbAI�Ak� ?�2XA��3A�j�?v�A4�A��@��b@��I@U��@���A��qA�7�A��C���A���A�m�A�mAVJA.�C��MA�5�A�#C��gA��A�Q�B��A���A�zA�l$A��A�{]A�%@�PA���C�h�A;'A�PSA� �Aߎ�Ad�,@��O@W��A`/AS}�A/�A ��A�n�A=icA�y�@KS�@��AF� A�
bA[B1A�^�A�x"B��AAu,Al�A?�A�@MA�H{?g A��A��D@�3�@�Y�@U��@�,QA�^A�(A�C��_A�xaA��A�%AV�UA-�C��mA�{�A��C���A���A� B�y         #         	      $   	   2         -                     	   #               &   .             4      $         	      =            q      5   ]         l   3                                                               3               !                  /      #         !   /      1      /      !               )            =      +   )         !   '                                                               !                                 %      !            '      1      +                     !            '      )            !   !                                 N�K�Np�rO��N�0O0|N��O�d�N�g�N�O�)�O���N�|�OT��O�O|�/N�N��N4%�N�^NgSO�)�O�O�XMO�fN��OO��O�9XO["pP'3�N��?P&	VN�x�O��O	d�M��_N��CNft�O��N�mN��N�,�O�_MN�rAO߉eO��O%�HN0�O�aBO�4�O�N��OkN� bN?J�O+��OA,NX�&N�mN�j8NR�9  �  �  �  �  �    �  �  %    �    y  �  _  R  *  s  n  &  �  �  �    �  �  7    N  o  7  �  �  �  �  �    c  f  %  :  �  e  �  �  �  �  *  �  �  !  �  |  v  �  H  �  �  �  ���7L�49X%   ��o�o;D��;�o<�C�;��
<���<t�;ě�<�1;�`B<T��<t�<T��<49X<���<e`B<���<�t�<��
<ě�<ě�=�P=��=\)<�<�h=\)<�='�=t�=o=C�=C�=H�9=�w=49X=8Q�=�E�=<j=L��=�\)=@�=Y�=T��=�o=u=ix�=y�#=q��=�%=��=��=���=�j=��=ȴ9����������������������������������������miintu���������tmmmm��������������������22345BGN[\gmlg\NBB52`]__grt�����tg``````#0<HLNNJF<10)#---/<HNUVYUQH<2/----�������������������������� 
!'38::80#
�"/5:HNMHB;/"����������������������������������������gghlrt�����������tgg#0<IUYWPI<0#�������������������������������'" )67==6)''''''''''zwv{������������zzzz366BOZYOEB@633333333��������������������z������������������z��������������������bUPIB<0$# #(0<IUXbb��������������������`__bhrt����������th`�����'+-474)������������������������)4N[g���x[N������������������������ )7C@AA;=;5):<BBJNV[agjig_[PNB::%)5BNX\^YNB5)����
!#%$#!

��cfhnt���ythhcccccccc����������������������������������������������������������


���������-),-/4<>HHOOOLH<:/--��������������������������$&$�����)6:BFHHDB?6.)"O`iot����������rl[OOHFFN[gt���������g[NH;954<HJR]abecaUHHA<;���������������������������
!##
����������*./.+)�������� !����).57:85)!!$/14451/(#��������������������,/<@HIKH=<;50/,,,,,,"/;DHJLKHC;1/#"������*63+�������������������������#
�������
#$%###���������������������

�����������������	�
��
����������������������������¿����������¿²«¦£¦²º¿¿¿¿¿¿�n�zÇÓÞàçäàÓÇ�z�n�h�d�e�n�n�n�n�[�_�c�g�j�g�g�g�[�N�M�N�Q�W�[�[�[�[�[�[�����������������������������������������B�N�[�g�i�g�g�d�[�N�B�>�6�9�B�B�B�B�B�B�F�S�x�������������x�l�S�F�:�-�*�-�.�:�F��������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E;��(�A�M�Z�f�z���s�f�M�A�(���������/�H�T�a�d�e�a�T�D�;�/�"�������	��"�/����*�6�7�C�E�C�6�*����������čĚĦĳĿ������ĿĳĢĚďčā�|ĀāĊč�;�@�G�T�`�i�j�`�`�T�G�;�0�.�-�.�3�4�:�;����������������������r�e�_�V�T�Y�f�r������	������������������"�.�;�G�T�U�T�G�G�E�;�.�"������"�"�ʾ׾����׾ʾľǾʾʾʾʾʾʾʾʾʾʽ��������������ݽ׽ݽݽ����ʼռּ��ּʼü������żʼʼʼʼʼʼʼʿĿ˿����#�,�!�����ؿοĿ��������ľZ�l�s�w�v�s�Z�M�A�4�(��%�(�/�4�A�M�Q�Z�(�N�Z�g�{���������s�g�Z�N�A�(�����(����ֺܺκֺ̺׺��������������ﺤ���������������������������������������s�������������������������s�o�a�]�e�s�����	�"�;�H�O�T�H�;�/����������������׿	��"�*�,�.�,�.�"��	�������������	�	�"�;�J�K�@�/���
��	���������������	�/�;�H�O�T�V�T�S�J�H�E�;�/�,�,�(�/�/�/�/�uƚƳ����������������Ǝ�u�h�\�\�h�u�I�U�b�b�n�u�{�~�{�t�n�b�_�U�Q�I�G�C�I�I�ѿݿ�������������ݿѿǿĿĿпѿm�y���������������y�m�f�`�T�J�T�T�`�e�m�3�@�C�L�U�S�L�J�@�:�3�3�3�3�3�3�3�3�3�3�N�Z�e�g�q�g�b�Z�N�A�<�5�2�5�A�L�N�N�N�N���������������������������|����������������'�3�7�9�8�'��������޹޹����������	��������������������	��"�-�/�1�/�#�"��	�����������������:�F�S�l�x�~�����������x�l�_�S�F�D�:�9�:�'�4�@�B�8�)�"�������ܻѻѻܻ����'�������������ߺ����������'�@�Y�r�z���{�Y�R�@�������������'�B�O�[�j�n�o�m�h�b�N�B�6�)�+�0�/�)�*�3�B�������������������������s�o�g�r�s�������/�<�H�J�H�D�<�1�/�.�#�"�#�(�/�/�/�/�/�/D�D�D�D�D�D�D�D�D�D�D�D�D�D�DsDqDxD�D�D�¿����������������¿²¦¨²¿��#�0�<�=�@�?�<�3�0�#���
���
������������������������������������������ؾʾ׾����	�����	�����׾Ҿʾ��¾ʽнݽ��������������ݽֽнϽн�F$F*F(F$FFFFE�E�E�FFFF$F$F$F$F$F$�����
�����
�����������������������`�l�t�y�{�y�|�~�}�y�l�`�S�L�L�M�Q�S�_�`EuE~E�E�E�E�E�E�E|EuEjEiEuEuEuEuEuEuEuEu�a�\�b�d�j�n�o�{ŇňňŇņŅ�~�{�n�a�a�a�zÆÇÓ×ÞßØÓÇ�z�w�n�k�i�n�o�y�z�zǮǯǭǧǡǔǐǌǔǔǡǮǮǮǮǮǮǮǮǮ H G $ n , ! &   F 7 1 u  5 6 R B @ / F X t m J C * M < [ 1 k X  , z R v = ? F l h ^ G ^ X G  8  O J u V : k k f 7 7  �  o  F  �  ~  �  S  �  +  �  \  �  �  >  �  P    ;  �  �  �  �    M  �  �  B  �  +  �  �  �  
  (  Z  �  �  )  4    G  �  #  $  �  }  P  �  �  >  �  \    t  t  �  �  '  &  m  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  Bj  �  �  �  �  �  �  �  �  u  b  M  7     	  �  �  �  �  {  [  r  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  \  %  �  W  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  v  p  k  f  `  �  �  �  �  �  �  �  �  �  n  N  +    �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �    p  b  S  C  2      �  �  �  w  j  [  I  3    �  �  �  �  e  /    �  �  X  U  �  �  L    �  �  �  �  �  �  �  e  0  �  �  V  �  �    �  %        �  �  �  �  �  �  �  y  e  K  0  	  �  �  ~  M  R  �  �  �  �      �  �  �  �  �  m  D    �  _  �  3  �  �  �  �  �  �  �  y  U  3       �  �  �  �  V    �  @  �    �  �  �  �  �  �  �  �  �  �  �  �  v  ^  F  !  �  �  �  �  �    ;  Y  o  y  r  W  .  �  �    ?  �  �  A  �  �    �  �  �  �  �  {  s  m  h  e  ^  Q  8    �  �  �  ]  '   �  P  Y  ]  ^  _  ^  Y  O  B  8  .  %    �  �  �  �  :  �   �  R  J  B  9  1  )  !                          	    �  	      '  )  *  '  #        	    �  �  �  �  \    Z  ^  b  f  i  m  q  o  k  f  a  \  X  R  K  C  <  5  .  '  0  E  O  V  [  \  b  i  m  l  a  M  /    �  �  �  J  �  p  &  "                   �  �  �  �  �  �  �  
  A  x  1  <  S  l  �  �  �  �  j  H  "  �  �  �    ,  �  N  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    |  {  z  y  �  �  �  �  �  z  i  ^  i  �  �  �  r  K    �  d  �  �  ?      �  �  �  �  �  �  �  �  �  l  E    �  �  �  �  �  �  �  �  �  �    t  h  \  N  @  /    �  �  i  *  �  v  �  n  7  c  ~  �  �  �  �  �  �  o  T  2    �  z  %  �  k    �  �  �    *  6  2      �  �  �  O    �  H  �  �    �  e  �  �  �  
          �  �  �  �  �  f  #  �  j    �  �  N  M  F  8  9  5  6  G  K  :    �  �  �  �  �  A  �  Q  �  o  Z  H  8  2  ,  $      
     �  �  �  �  �  �  5  �      2  6  0  %      �  �  �  �  �  L    �  �  A  �  �  {  �  �  u  `  G  *  	  �  �  �  w  Q  ,    �  �  �  �  �  �  e  ~  |  w  z  �  u  Y  9    �  �  �  �  T    �    �     �  �  �  �  �  �  �  �  j  ;    �  �  H  �  �  9  �  p    �  �  �  p  ]  J  4    	  �  �  �  �  �  �  c  A    �  �  �  �  �  �  �  r  R  2    �  �  �  �  Y  3    �  �  �  j      �  �  �  �  �  �  �  �  �  �  |  j  X  F  4  "     �  +  K  T  \  b  a  O  &  �  �  {  :  �  �  8  �  �  �  R  h  f  d  a  ^  [  X  V  L  ?  2  %      �  �  �  �  �  z  a  %    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  6  :  :  6  .  !    �  �  �  �    d  B    �  �  E    �  
�  G  �  :  �  �  �  �  �  y    �  2  
�  	�  �  L  �  :  �  ]  c  d  b  _  Y  S  M  F  .    �  �  w    �  f  0  �  u  r  �  �  v  Y  "  �  �  `  ]  d  �  �  �  �  d  
  �  �     	�  
i  
�  Y  �  �  �  �  z  S  .    
�  
�  
:  	�  �  e  �  6  �  �  �  �  �  �  �  �  �  �  �  �  �  m  S  9  ,  #  A  c  �  �  �  �  �  �  �  �  w  _  C    �  �  t  8  �  �  v  1  *    �  �  �  =  �  �  �  �  �  �  ~  5  �  5  
e  	u  s  �  O  �  �  �  �  �  �  g  7    �  �  8  �  x    �      r  �  �  �  �  �  �  u  W  1    �  �  r  >  	  �  �     �  �  !             �  �  �  �  �  �  �  k  3  �  }  �  W   �  �  �  �  �  �  �  �  �  �  e  E    �  �    D    �  x  (  |  ]  F  r  �  �  x  j  ]  P  W  r  u  i  \  P  B  4  $    v  V  8    �  �  �  w  =     �  w  ,  �  �  B  �  �  (  �  �  g  G       �  �  �  m  1  �  �  w  9    �  �  ]  "  �  H  G  E  =  2  !    �  �  �  q  7  �  �  /  �  i  	  �  I  �  �  |  b  J  2      �  �  �  �  �  �  �  t  [  @  #    �  �  z  S  ,    �  �  q  ?    �  �  �  �  �  d  @  7  o  �  z  <  
  �  �  p  9  �  �  �  >  �  �  C  �  >  �    {  �  X  1    �  �  �  �  ^  6    �  �  �  T    �  �  M  