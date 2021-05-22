CDF       
      obs    N   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���E��     8  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P���     8  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �}�   max       <t�     8      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @F�33333     0  !T   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v�fffff     0  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @O�           �  9�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @��@         8  :P   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �Q�   max       ���
     8  ;�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�]    max       B4+�     8  <�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�Ye   max       B4}�     8  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�Օ   max       C���     8  ?0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >B��   max       C��     8  @h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �     8  A�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     8  B�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?     8  D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�	*     8  EH   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��3���   max       ?�|�hr�     8  F�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <o     8  G�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�=p��
   max       @F���
=q     0  H�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v�fffff     0  U    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @O�           �  aP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @�l�         8  a�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?B   max         ?B     8  c$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��,<���   max       ?�|�hr�     `  d\         
   !                     "         7   $   
       	                        	      I   '                           '                	   	               
   	      5                
            1   �               	            
      4   	            N�Q�N�ּN͚PO��~Om�O��O�q�O���N�)N�PQ��NM�NG{P���O��N�#�O�c�O�O�+N�,O�@3N��)Om��O�FOL��N�i�O2�P�@P)f?N�U�N�4rN5��NV:Oh�OP�DN��"M���O�xO�^�O� mO�y�N�a>N1��N]<N��OĆO�N�O)VjO0�Ny3AOc|�P.4LN���O2(�O�.uN��0O-
N��N�<PO�vxO��PzN��0O*c�NB�O0�cN�`�N�8N�z�N��O *!O�}O�1KN���N)��OPu�OyKNW�x<t�;ě�;�o;�o;o�o�o��o��o���
�ě���`B�t��#�
�D���T���e`B��C���C���t���t���t����
���
���
��1��1��1��j��j�ě���������������/��`B��`B��`B��`B��h��h��h��h��h��h���������C��C��C��\)�\)�t��t���w�#�
�#�
�#�
�'',1�,1�,1�,1�0 Ž8Q�8Q�P�`�T���aG��aG��aG��u�u�y�#�}������������������������������������������������������)0<I^ns{�{vmbUI<4&$)��������������������&5BK[gmpmh`TNB50)?BGO[hq{}}zqh[OB>=??��������������������ggqt|����{xtnggggggg\acnz�zna\\\\\\\\\\{}���������������{u{�����������������������


���������������0<n{������{UI0�������		������������

��������������������������������������������������;AIbuy{���nbUOI>@<@;&)15ABMNYXNBBA5))!&&`gt�������������~tg`LOW[chrtuth`[VOKLLLL=BHN[gt�������tgTB9=&+16COY\e]\VLC=6*'$&s{��������������uops�	������������������
#&#
����w����������������xswxz�����������~x������	������������
#+,#

 ggqt����tqggggggggghnrz�����zzvpnhhhhhh)BIOPUTOB?6)mrz������������zmihmSTamprvwwmfa^ZTSSSSS05BBGCB>550000000000ez��������������znde�����������������/8BN[gpsrrpjg_[N>++/[dt�������������g[X[����������������������	�����������������������������������

 �����������������������������)5BN[gpqpg[NB5,)&%,)#%/<EHLMMHD</#��������������������ST^acddebaTRPQSSSSSShmuz�����������zmceh���������������X[agpt������ytmhig[X)6BO[``[YROB6.)#��
#0<MTU<0(%#�����"#%,05<=??<60.*(#! "x{������������}{rppx),/0/)����������������������)Ohtwthc[O6������
#&*,.#
���������"&%���������������������������� �������

������������#(/5<HDC?95/# #��������������������6<>HSUZ[XUH<:6666666����������������������������������������������������15BNY[VSPNJB?5520011gjt����������yumg\]gtt�������ttpttttttttB6)#&()676BBBBBBBBBB�����
!##
������>EMO[dhljhec_[UOB:>>�������������������������������ûлܻ�ܻػлû��������������U�R�M�H�G�B�H�U�a�h�k�a�_�a�b�a�U�U�U�U����¿²¦¥¦²¿�������������������˼��������������ǼʼѼ���������ʼ����������������*�4�D�A�4�(�������s�h�Z�M�K�[�f�s���������������������m�d�`�U�U�Y�m�y�����������������������m����ųŭţŭŹ���������
��&�&�����������������������������	��	� �������������U�K�H�E�F�H�U�X�Y�W�U�U�U�U�U�U�U�U�U�U�ѿ����m�Z�I�G�T�j�y�����Ŀտ���!�����	��������	�� ���	�	�	�	�	�	�	�	�	�	���������������������������������������������s�N�B�6�:�E�Z�s����������������� ��y�u�k�P�J�H�T�`�y���������������������yŠŘřşŠŭŶŹž��ŽŹųŭŠŠŠŠŠŠ�������s�f�Z�J�S�s����������ӾܾܾӾʾ����	���������	��"�.�;�=�;�2�.�"���-�!��!�-�:�F�Z�_�j�g�l�s�x���x�l�_�F�-�m�j�a�_�a�a�m�q�z�����������������z�m�m�z�x�p�a�T�J�T�a�z�������������������}�z������|���������������������������������������������	���!�!���	�	�������ܾӾӾ׾ھ�����	������	�����������������������ʾ־޾���߾׾ʾ��׾Ҿ׾߾����	���	�����׾׾׾׾׾׿Ŀ������������������������ĿƿǿƿɿͿĺ����x�\�^�������/�:�K�O�N�K�!���ɺ��������������������*�2�5�8�6�/��������������*�0�6�9�<�6�4�*�)��������������������������������������������ŇŇ�{�x�s�{ŇŒŔŕŔŇŇŇŇŇŇŇŇŇ�����������ùϹܹ�ݹܹϹù��������������F�?�B�N�S�[�_�x�������������|�x�l�_�S�F�����������������������
����
�������������������������������������������������y��������������������������������������������������Ŀѿݿ��������������ѿ�àÓÏÌÇ�|�s�q�Óâìù��������ùìà�)�����������)�5�B�F�N�Q�S�N�B�5�)�ϿĿ������������Ŀѿݿ���� ������ݿϿ`�[�[�Y�`�h�m�y���������������y�y�m�`�`F=F:F3F=FJFVF_FVFUFJF=F=F=F=F=F=F=F=F=F=�����������ɺ˺Ӻɺ����������������������������������ɺԺֺݺֺҺɺ�������������FFFFF$F1F=FEFJFVF\FXFVFJF=F1F$FFF������������*�6�A�E�H�F�@�6�/�*����������������	���"�)�"�%�"���	�����	��	���!�"�/�;�>�H�I�H�;�3�/�"��g�c�Z�Q�Z�g�s�������������s�g�g�g�g�g�g�(���
�����#�(�5�8�=�>�@�C�M�A�5�(�f�_�`�r�������ּ���������ּ����f�)�$����!���)�B�O�V�[�e�[�O�B�=�5�)�4�(������"�(�4�>�A�M�Q�K�P�M�J�A�4���������������������̽ݽ� ���޽ݽнĽ�������(�2�4�A�M�V�Z�\�Z�M�A�5�(������������'�4�@�M�M�I�@�>�4�'���h�\�[�O�G�K�O�[�h�t�~�|�t�m�h�h�h�h�h�h�g�b�]�g�n�s��������������s�g�g�g�g�g�g���z�y�t�v������������������������������ED�D�D�D�D�EEE*ECEPEiEeE^E\EPECE*EE������������'�M�Y�s����}�l�M��ìãàÞàéìù��������������ùìììì�����������������)�6�=�9�6�)�#���²¦¬²¸¿������¿²²²²²²²²²²E~EuErEuE~E�E�E�E�E�E�E�E�E�E�E�E�E�E�E~����������������������������������������FFE�E�E�F FFF$F)F.F,F$FFFFFFF�����������������������������������������������������ɼļ��������������������������
���������������������
����"�#��H�@�?�A�H�U�a�n�zÅÇÉÇ�}�z�n�a�U�H�H�;�/����$�/�H�U�z×ÛÔÇ�z�n�[�U�H�;ĿĿĸľĿ��������������ĿĿĿĿĿĿĿĿ���������������������������������������뻷�������������������ûлܻ��ֻܻлû��Ϲù����������ùϹ׹ܹ����������ܹ���������$�$�$������������������� L v < 8 R 3 + U h r X 8 b V 2 Q E : E 0 3 7 R L K \ h N g a C L ~ H . ] W & O 5 : . = ? 8 D 6 " 7 w C S U > M y , 8 , v W Q H T H l W  > i \ D L " N ' : >  �  �  �  d  u  _  #  �  [  f  �  /  T  R    �  �  A  =    �  �  %  f  �  �  �  �  x  Q  �  c  �  �  �       �      �    G     �  I  0  h  &  �  �  s  =  �  	  	  C  �  �  �  K  �  �  }  i  �  �  �  �  =  �  V  �  �  _  �  :  o�o���
�ě���h�T����o��C���/�ě��#�
��w�#�
�D����o�D����j�<j���ͽt�����w���49X�+�'���/����y�#������h���L�ͽ@��\)�o��+�]/�y�#�ixս�P��P��P�']/�<j�t��',1�T���� Ž',1��C��0 ŽH�9�@��<j�}󶽸Q�Q녽T���u�e`B����T���y�#�ixսq���}󶽑hs��"ѽ�%��C����T��{��\)B��B �B��B&��BJB<Bc7B��B	٭B�3B*�)B2B&�B&�B��BYOB!7FB�B'_BrWB+Bi�B	m�B0��B4+�BeeB/%B"�B�B9�B6B	��B)"B�eB ^�A��PByB�<BPBo�B
�2B�wB��B ӜB$HB�B�BP�B �A�] B `8B,z�B	�!B4B%�B%��B)[�Be#B�#B^qB�B�B�XB��B�)B�B ��BD�BydB�MB�GB�[B
tyB
UB��B$�B�4BfBBeYB ��BͩB'89B@@B��BJ%B?B	�MB�yB*��B>NB@rB&��B>IB�tB!?5B?�B'z�BbB
��B?�B	R�B0�B4}�B>�B��BFgB �B}�B@�B	��B�lB��B A�~�B��B�1B��B��B
�xB �Bg�B �yB$B�B?�B��BM%B=�A�YeB �B,�B	�pB�1B$�"B%��B)@)BC�B\^B?�B?�B?�B�vBB�BH$B7DB B�B@�B��B�HB��B�B
؊B
aqBM@B$E�B>�BR�@��8AşsA���A �A1ӨAD(Aop�A�G[A�A�MAw)VA��{A��1A��AnxA���AII_A\�@�\A�i�A��|@�J�AZOAXwAO�9AWd	Au�@?sB<EA���A���A�3>�Օ@���A�PA�"WA�F�Az(MA�{gA�n�A|��Al}�C���@-�@2nC�ΥA��_A[
�A��A��A���@��lA�ʀA8NA%�A9�@ɺjA�"�A���A��KC���@öCA���AԸA��C�&�@��JC��#A�+�@�F�A�ڻA�^�A�m!A㬄A��o@�">��_B	�@��hA�`�A�y�@�K!A2��ABnAoBA���A��
A�pUAt�A��A�SA��jAn�=A���AL��A[Ծ@tӀA�f�A��e@�#^AZߧAW�AO	AU,RAuܰ@;v�B�A�qTA�r�A�>��Y@�/A�}A�u2A��|Ay{�Ã�A�8A|��AlOC��@,��@3�5C��3A�y�AY��A��_A�bvA���A1kAح"A6�KA$�$A<W@�sA��A���A�yVC���@��A�]NA�u3A�V�C�'d@�E�C���A��v@���A��A�u�A�0A�e�A�{�@��`>B��B�            "                     "         7   %      !   	                        
      J   '                           '      !      	   	   	                  	      5         !                  2   �               
         	         5   	   	         	                                 ;         A         '      !      !                     =   -                                                                     1         %               %      -                                 !                                                1         ?         %                                 =   )                                                                     1         #               !                                                      N��zN���N͚PN�FJN�E�O��OK��O+��N�)N�PJ�NM�NG{P�	*O�q]N�#�O�s%N� O?��N��N�3�N��)O��NO!��N�i�O2�Pt4�P�&N�U�N�4rN5��NV:OP�@O>�N��"M���O>��OxD�Ox��O���N�a>N1��N]<N�_N��O�wDO)VjN�	�N5p+ORZBP.4LN���O2(�O���N��0O-
N��N�<PO�hN�ϭOoh�N��0O/\NB�O�+N�`�N�xN�z�N��O *!N�=�O!�0N���N)��OPu�OyKNW�x  �  �  �  �    �  L  �  �  k  D  �  �  T  �  �     W    �  q  �  a  9  �  �  .  �  c  !    q  �  y  �  8  K  �  �  �  �  �  �  =  �    �  ;  1  �  n  �  S    6  x  ~  f  �    	\  8  �  �  Y    '  �  �  a  �  w  t  !  G  �  �  [<o;��
;�o�49X��o��`B���
�D����o���
�T����`B�t��D������T���u��t����
���
��h��t���/��9X��j��1��1����/��j�ě�����������`B��`B��`B��`B�,1�����o��h��h��h�o��w�����o�t��\)�C��\)�\)�#�
�t���w�#�
�#�
�,1��o���ͽ,1�49X�,1�8Q�0 Ž<j�8Q�P�`�T���ixս���aG��u�u�y�#�}������������������������������������������������������:<IU[begdbUIA<94::::��������������������45BN[^ghgc\SNB;45:54DOW[hmwzzwtsh[OJCCCD��������������������ggqt|����{xtnggggggg\acnz�zna\\\\\\\\\\y����������������{y�����������������������


����������������0En{�����{U<0��������		������������

��������������������������������������������������GIUbnrvvvnbUIEABB?>G$))458BKNONNB5,)$$$$�����������������LOW[chrtuth`[VOKLLLLNNU[gt~�����tg_[ONNN(*-36ACOQ\YSJC86/*&(����������������yvw��	������������������
#&#
����z����������������~yz�������������������	������������
#+,#

 ggqt����tqggggggggghnrz�����zzvpnhhhhhh)BGORTQOBB6)msz�����������|zmjimSTamprvwwmfa^ZTSSSSS05BBGCB>550000000000����������������~|�����������������15:BN[gnrqqog][NB/-1[_gt������������og_[����������������������	�����������������������������������

�����������������������������&).,5BN[gmpng^NB5-'&#%/<EHLMMHD</#��������������������RTTaccdaaTTRRRRRRRRRimvz�����������zmefi���������������X[agpt������ytmhig[X)6BO[``[YROB6.)#����
#0<GON<0&#
���"#%,05<=??<60.*(#! "x{������������}{rppx),/0/)����������������������)4O`lh`[OB6���
 ###"
	������
�������������������������������

������������!#&/9<=@=<73/#!��������������������7<?HQUZZWUH<;7777777����������������������������������������������������35BNUXTQONB753113333z��������������zvutztt�������ttpttttttttB6)#&()676BBBBBBBBBB�����
!##
������>EMO[dhljhec_[UOB:>>�������������������������������ûлܻ�ܻػлû��������������U�S�N�H�G�C�H�U�a�f�i�a�]�a�a�a�U�U�U�U����¿²¦¥¦²¿�������������������˼����������żʼּ�����ּܼʼ����������������������$�'��������s�r�c�Z�V�Z�f�s�u�����������������u�s�m�c�`�^�a�m�y�����������������������y�m������ſŻ���������������������������������������������	��	� �������������U�K�H�E�F�H�U�X�Y�W�U�U�U�U�U�U�U�U�U�U��ѿ����y�d�_�`�y�������ĿϿ��������	��������	�� ���	�	�	�	�	�	�	�	�	�	�����������������������������������������������S�A�9�=�I�s����������������������y�m�^�W�T�Y�`�m�y���������������������yŠŘřşŠŭŶŹž��ŽŹųŭŠŠŠŠŠŠ���s�d�M�U�s������������Ҿ۾۾Ҿʾ���������������	���"�,�-�"��	�������������!� � �#�*�-�:�F�S�_�e�i�u�l�_�S�F�:�-�!�z�n�m�a�`�a�c�m�y�z�}�����������z�z�z�z�����z�o�m�k�m�z������������������������������|���������������������������������������������	�������	������������߾׾վ־׾����	������	���������������������ʾѾ׾ھݾݾܾ׾ʾ����׾Ҿ׾߾����	���	�����׾׾׾׾׾׿Ŀ������������������������ĿƿǿƿɿͿĺɺ����~�p�j�l�����ɺ�.�F�F�A�!���ֺ��������������������$�-�4�6�4�,��������������*�0�6�9�<�6�4�*�)��������������������������������������������ŇŇ�{�x�s�{ŇŒŔŕŔŇŇŇŇŇŇŇŇŇ�����������ùϹܹ�ݹܹϹù��������������S�F�A�C�P�S�^�l�x������������{�x�l�_�S���������������������
����
�
�������������������������������������������������y����������������������������������������������������Ŀѿݿ�������ѿĿ�àÓÐÍÇ��u�vÃÓÝìðø������ùìà�)�#�����������)�5�E�N�P�R�N�B�5�)�ݿѿȿ¿������Ŀѿݿ���������������ݿ`�[�[�Y�`�h�m�y���������������y�y�m�`�`F=F:F3F=FJFVF_FVFUFJF=F=F=F=F=F=F=F=F=F=�����������ɺ˺Ӻɺ����������������������ɺź��������ɺϺֺںֺ˺ɺɺɺɺɺɺɺ�FFFFF$F,F1F2F=FHF=F:F1F$FFFFFF��������������*�6�?�C�G�E�?�6�*�������������	���"�)�"�%�"���	���"��	��	���"�"�/�;�=�H�H�H�E�;�/�%�"�s�g�g�\�g�s�������������s�s�s�s�s�s�s�s�(��������&�(�5�6�<�=�?�C�K�A�5�(�f�_�`�r�������ּ���������ּ����f�)�$����!���)�B�O�V�[�e�[�O�B�=�5�)�4�(������"�(�4�>�A�M�Q�K�P�M�J�A�4�Ľ����������������������ǽݽ����ݽнľ�����(�2�4�A�M�V�Z�\�Z�M�A�5�(������������'�4�@�M�M�I�@�>�4�'���h�\�[�O�G�K�O�[�h�t�~�|�t�m�h�h�h�h�h�h�g�b�]�g�n�s��������������s�g�g�g�g�g�g���}�y�v�x������������������������������EEEEE*E7ECECEEEPEXEREPECE7E*EEEE������������������'�2�=�?�5�'��ìãàÞàéìù��������������ùìììì��������������)�6�9�6�6�)�����²¦¬²¸¿������¿²²²²²²²²²²E�E�E�EwE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�����������������������������������������FFE�E�E�FFFF$F(F,F'F$FFFFFFF�����������������������������������������������������ɼļ��������������������������
���������������������
����"�#��H�D�A�D�H�U�a�n�z�}Å�z�w�n�a�U�H�H�H�H�/�)�*�/�<�?�H�U�a�c�n�zÃ�{�z�n�a�U�<�/ĿĿĸľĿ��������������ĿĿĿĿĿĿĿĿ���������������������������������������뻷�������������������ûлܻ��ֻܻлû��Ϲù����������ùϹ׹ܹ����������ܹ���������$�$�$������������������� J p < - / 9 ( L h r Y 8 b X - Q D ' *     7 B F F \ h J e a C L ~ F 1 ] W  R 4 8 . = ? " C 3 " < w @ S U > J y , 8 , q I  H N H j W  > i \ > S " N ' : >  �  �  �  �  �  [  �  �  [  f  �  /  T    \  �  �  �  �  �  �  �  N    r  �  �  B  #  Q  �  c  �  �  �       �  �  �  Z    G     6  �    h  	  d  �  s  =  �  �  	  C  �  �  $  �  �  �  9  i  M  �  �  �  =  �    }  �  _  �  :  o  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  ?B  �  �  �  �  �    j  R  :  "    �  �  �  �  \  <        u  ~  �  {  n  ]  I  4      �  �  �  �  �  �  _  2    �  �  �  �  �  �  �  �  �  o  Z  G  5  %              %  �  �  4  e  �  �  �  �  �  �  �  �  �  �  O    �  ,  �  �  �  �  
        �  �  �  �  �  �  f  7  �  �  h    �  �  k  z  �  �  �  �  �  �  �  �  �  �  �  q  Y  L  q  {  r  ^    /  >  H  L  I  A  9  0  &      �  �  �  �  l  ;     �    6  O  ]  k  {  �  �  {  k  S  1    �  s  '  �  �  J  �  �  �  �  �  �  �  |  w  q  k  b  U  H  <  /  "    	   �   �  k  k  l  l  l  k  j  i  j  m  o  r  g  R  =  '    �  �  �    0  7  ?  D  ?  .    �  �  �  i  +  �  �  }  V    �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  y  t  p  k  f  b  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  E  G    �  �  Z    �  z  ,  �  t    �  �  �  s  B  �  &  �  �  �  �  �  �  �  �  �  o  >  �  �  p    �  /  �    �  �  �  {  b  H  (    �  �  �  �  �  h  N  0    �  �  �  �  �     �  �  �  �  �  �  �  �  �  �  �  �  �  y  ;  �  F  ^  2  B  Q  U  R  P  N  L  K  J  A  2  "  
  �  �  �  �  �  v  �  �  �    �  �  �  �  �  �  �  �  �  u  Q  (  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  n  Z  E  0    �  �  �  �  �  �  �      8  T  g  q  p  i  Z  H  1    �  �  �  �  �  }  w  m  a  P  ,    �  �  p  <    �  �  d  +  �  �  �    7  H  V  _  _  T  D  .    �  �  k    �  t    �  �  4  7  8  8  /  $      �  �  �  �  �  g  @    �  �  S   �  �  �  �  �  �  �  �  �  �  �  �  m  J  %     �  �  I  �  i  �  �  �  �  �  �  �  p  Q  -    �  �  �  S  '    �  �     .      �  �  �  �  �  �  �  �  �  �  �  �  �  }  G     �  u  �  �  �  �  �  z  k  T  5    �  �  u  2  �  �  �  �   �  8  ^  c  \  M  4    �  �  b    �  t  ]    �  �     x  �  !        
    �  �  �  �  �  �  �  p  O  &  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  }  s  j  ]  Q  E  q  e  Z  O  C  8  ,      �  �  �  �  �  �  �  �  �  �  w  �  �  �  �  �  �  �  �  �  �  �  �  |  v  p  j  B    �  �  u  y  u  h  S  8    �  �  �  r  A    �  �  S    �  �  �  �  �  �  �  �  �  �  �  �  �  x  `  H  -    �  �  E  �  (  8  -  "      �  �  �  �  �  v  [  C  +    �  �  �  I    K  ?  2  &      	    �  �  �  �  �  �  �  �  w  b  N  9    4  S  c  m  �  �  �  �  �  �  �  O    �  S  �  �  
  7  �  �  �  �  �  �  n  ?      �  �  �  [    �  �  O  �    �  �  �  �  �  �  t  T  0    �  �  d    �  E  �  _  !  �  �  �  �  �  �  ~  k  Z  F  ,    �  �  a    �  u    �  v  �  �  �  �  �  �  �  �  �  �  s  _  J  2      �  �  �  �  �  �  s  c  S  C  2  "                �  �  �  �  �    =  D  K  N  P  Q  O  M  K  H  D  >  8  .  $      �  �  �  }  �  �  �  �  �  �  �  �  �  m  J  #  �  �  �  n  <  
  �  �  �  �  �  �            �  �  �  z  I    �  �  �  V  �  �  �  �  �  x  u  f  Q  ;  #    �  �  �  X  (  �  z    ;  4  -  '  "            
    �  �  �  �  �  �  |  _  ,  .  /  "    �  �  �  �  �  }  b  N  ?     �  �  �  u  N  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  n  h  N  /    �  �  �  �  k  ?  	  �  �  ?  �  s  �  Y  �  }  ]  6    �  �  �  K  
  �  �  K  �  �  ]    �  �   �  S  Q  O  M  G  A  :  3  -  &  -  C  X  _  J  5      �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  `  )   �   �    *  4  3  '    �  �  �  �  o  M  &  �  �  J  �  i  �   �  x  s  n  g  Z  M  <  *      �  �  �  ~  U  1      %  :  ~  }  |  v  n  b  U  F  5  !    �  �  �  �  v  P  *  	   �  f  ^  W  L  :  '    �  �  �  �  �  ~  b  ^  x  �  �  �  �  �  �    o  b  U  H  =  3  )             �  �  	    $  �          �  �  �  �  _  :  	  �  �  �  `  -  �  �   �  $  o  �  �  �  �  �  	   	N  	[  	I  	   �  �  G  �  g  �  �  T  �  /  0    �  �  (  8  "  �  �  5  �    T  Y  �    
0    �  �  �  �  �  �  �  �  n  K  #  �  �  �  F  �  �    �    �  �  �  �  �  �  �  �  w  ]  ?  !    �  �  �  �     #  B  Y  L  @  4  '      �  �  �  �  �  r  b  X  T  a  o  o  R  �              �  �  �  �  �  q  [  1  �  �  �  9  �  '      �  �  �  �  �  �  �  �  �  w  e  D    �  �  E  �    �  }  o  _  L  6    �  �  �  p  6  �  �  W    �  R  �  �  �  �  �  �  �  �  �  �  �  }  a  ?    �  �  f    �  y  a  �  �  �  �  ~  ^  >    �  �  �  �  |  Z  8    �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  Z  1  �  �  �  z  Z  :  q  u  w  w  u  q  i  ]  I  -    �  �  ;  �  �  M  �  a   �  �    c  �  �    J  o  t  i  J    �  �  8  �  e  �  �  +  !        �  �  �  �  �  �  �  �  �  �  �    ,  F  `  z  G  9  +    �  �  �  �  u  '  �  �  �  �  �  �  n  ]  P  D  �  �  �  t  _  G  (    �  �  �    j  O  2    �  �  x  ?  �  �  z  ^  A  <  C  V  O  5    �  �  �  O    �    �  	  [  E  0      �  �  �  �  �  �  �  u  b  P  =  *  �  ]  �