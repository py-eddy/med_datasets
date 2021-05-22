CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�Z�1'     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P¾�     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �� �   max       <D��     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Y�����   max       @F���
=q       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v��\(��       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @O@           �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ʩ        max       @��          4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �@�   max       �o     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�s)   max       B4�     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4[�     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�K   max       C��J     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >G\g   max       C���     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P���     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�:)�y��   max       ?�쿱[W?     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��9X   max       <D��     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�=p��
   max       @F�z�G�       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v��\(��       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @O@           �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʩ        max       @���         4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?C   max         ?C     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���vȴ:   max       ?�쿱[W?       cx               >                                    6                                          "   	   E   (      $                        *      �   	      
   
   +   	   -               $      
            #   	            	      %         4         N��mN�8CN�%�NVO�P¾�O���O=e�O���N+��N�PM���N���O�3�O���N��qN���P���Nv�O��N��<Oõ�O=5ON�O$��OEW@OkzQN
s�N�@%OJ#�O@m3O䩄N��P��iO�WxObT<O���N�3�N��NMhN:.O��iO��N��\P%O�_]O��jN4��O��O
��OF�PPNN�wxP*��N7'�OH�Nl�BN�CHOՁ�Nu�O(��Nc[�Oɫ8N�m�P��N�_O�N���N��ND2Nx�O=>N�w7O�O��N��O��OH��<D��;��
;D��:�o%@  �o�o�D���D�����
��`B��`B�o�o�#�
�#�
�49X�49X�D���T���e`B��t���t���t����㼣�
��1��9X��9X��j�ě��ě����ͼ�������������������������/��/��/��`B��h�������o�o�o�+�+�+�t��t���P��w��w��w��w�''0 Ž0 Ž<j�<j�@��D���P�`�T���T���]/�e`B�e`B�}�}󶽰 ��������������������������������������������������������������������������Ha�����()�����aH?H>BFO[_hk�~zth[O@>?>$0@IUbfnsnjbUIF<;40$��������������������Zaanz��znaZZZZZZZZZZggost����ytggggggggg������������������������

���������)5BHMNS]_[VB5)gyz�������������|nmg���

���������mnz}����zynjkmmmmmm���0<b{�����{U<��������������������������������������

��������������
�����������������������

���������<BCLO[dhptwthc[OGB8<BNO[gt�������utgYNCB���������
 ("
����#/2<<</#").46COS\e\[ULC962*')s}�������������uklps<=IUbossrnibUKIB;=?<W]gt�����������g[VTW��������������������wz�����������������wz�����������znieddnzmoz�����������ztmhfm)-3BN[ktvsqlg[NB2))��
#%+/#
�����uz�����zxuuuuuuuuuu��������������������fgnt����tttgffffffff�������� �������6GOTWOJB>6-��
"
����������������	�����������)*5BFgnsumgNB5,)%"))�����$(& ����������	
�����������������������������������������������������T[t�������ztjgf^[XQT���
0UbljbI<*&#���!#&.06<>@@?<0,)&#! !���������������TT_abcdca\TTRRTTTTTTimoxz���������zumefi%/<EHJH?</)$%%%%%%%%))..,)���
#*.&
����������

������������mov{������������{wnm�����������������������)5BOZkid[O6��������������������������	'095#
����������� ��������������������7<=HSU[]ZUH<:7777777������������������������������������}w�
&)/4/#
���!#(/6:940/#!  !!!!!!05BNZYVRPNKB?5520000gt����������{tojgc[g��������������������?BDLO[dhmjheb^[XOB>?����������ۻ����������ûлܻ�ܻڻлû�������������¦£¦²¿������������¿²¦¦¦¦¦¦����������������,�4�=�4�(�����U�Q�H�G�G�H�U�a�f�j�a�a�U�U�U�U�U�U�U�UƬƛƌƮƼƵ�����$�J�T�D�C�9�$�������Ƭ�m�c�`�U�M�T�X�`�y���������������������m���������������ƼʼԼ��������ּ̼�����ŵŭůŹ������������'�+���������U�I�H�D�E�H�U�Z�Z�W�U�U�U�U�U�U�U�U�U�U���������������������������������������ǈǇ�{�q�{ǈǔǖǟǔǈǈǈǈǈǈǈǈǈǈàÝàêìï÷ùü����������ùìàààà�m�_�^�Z�\�a�m�r�z�������������������z�m�x�m�a�T�K�T�a�t�����������������������x�����������������������������������������/�.�%�/�7�;�H�R�T�U�T�K�H�;�/�/�/�/�/�/���������H�=�8�>�Z�s��������������������G�=�;�:�;�<�G�K�T�V�T�P�G�G�G�G�G�G�G�G�����s�^�J�W�o�|���������ʾҾؾҾʾ�����ŠŔŖŞŠŭŮŹ��������żŹűŭŠŠŠŠ�y�w�m�k�N�H�H�`�y���������������������y����׾˾Ѿ׾����	������	��������������*�+�-�8�=�?�<�6�/�*����������}�z����������������żǼ��������������������	����"�"����	���Ŀ������������������������Ŀ̿̿οҿɿ��6�+�*�4�6�B�B�C�C�B�B�7�6�6�6�6�6�6�6�6������־־׾߾������	������	���������������������ʾվ޾�����׾ʾ��:�-�$�%�&�-�:�F�S�W�_�c�j�x�{�x�_�S�F�:�ѿĿ������������Ŀѿ����������ݿѿ`�W�W�`�c�m�y���������������y�m�`�`�`�`�ɺ������o�k�����ɺ��:�S�`�S�F�!���ɿ����������ѿ���������������ݿѿ��������������������������
�����
��������4�)��� ��������)�5�B�J�R�T�S�N�E�B�4���������������������������������������׹ù��¹ùϹܹ�ܹԹϹùùùùùùùùùür�p�f�c�_�f�r�{�����������r�r�r�r�r�rŇŁ�{�u�u�{ŇŉŔŔŔŋŇŇŇŇŇŇŇŇàÓÌÉÇ�z�t�u�zÉÓàäó������ùìà�F�<�9�G�S�\�l���������������|�x�l�_�S�F�������������ɺֺ޺����ֺɺ����������@�5�'���
���'�L�~�������������r�e�@������������*�6�C�F�K�I�B�6�.�*���������߻�������A�M�p�Y�@�'���F=F:F1F=FJFVF`FVFUFJF=F=F=F=F=F=F=F=F=F=FFFFF$F1F=FEFJFVFXFVFOFJF=F1F$FFF�"���	���	��"�+�/�;�=�H�K�P�H�;�/�"������)�6�B�O�W�[�c�[�V�O�B�=�6�)������������|���������������ݽ�����ѽ�������(�1�4�A�M�Z�a�b�Z�M�A�5�(���f�e�r�����������������ּʼ�����f�g�f�Z�R�Z�g�s�������������s�g�g�g�g�g�g�(�����	�
��� �(�2�5�:�?�A�H�A�5�(����������	������	�����������������O�D�O�O�[�h�t�y�x�t�h�[�O�O�O�O�O�O�O�O�4�!��#�)�B�N�t££�u�g�N�B�4²§§²µ¿������¿²²²²²²²²²²�'�����������'�4�@�D�J�M�M�B�@�4�'�g�b�^�g�l�s�z���������s�g�g�g�g�g�g�g�g���|�w�v�x�����������������������������������������������������������������������x�^�W�l�z�������ûܻ���׻û��������xìåäìñù��������������ùìììììì�������������� ����)�6�7�5�)����FF E�E�E�F FFF$F,F/F.F$FFFFFFF�������������������������������������������������������ʼӼʼ¼�����������������ĿĿĿ��������������ĿĿĿĿĿĿĿĿĿĿEED�EEEE*ECEMEPEUEYE\EbE\EXEPE7E*EE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��H�=�<�@�H�U�a�n�zÅÇÉÇ�|�z�n�a�U�H�H�6����#�/�<�U�zÍÒÐÇ�t�n�a�W�U�H�6�����	�����������������������������Ϲǹù����������ùϹֹܹ���������ܹϽ.�!���������!�.�8�:�G�R�X�S�G�:�. R H B ` p * = T y ` } I 6 C h 8 U ` 8 \ 2 T g * L L | J P 8 8 0 R   1 : H b Z C H R ( = < 8 = = L 2 V } Q { E 1 : 3 < ) - u ? I 6 O   : j B U c I P R 4 [    �  �  1  �  1  (  �  �  �  C    �      �  �  \  L  �  �  �  ,    h  �    }  9  �  �      �  �  �  z  �  .  n  H  ?  0  �  �  o  _  H  4  :  �  �  &  V  �  i  �  �  �  �  i  p  �  �  �  �  e  �  �  �  /  �  �  ]  �  7  &  ֻo��`B��`B�#�
�y�#��o�����ͼt���`B�49X�D�����ͽC��u�D������u�,1���ͽ49X��h�����'�`B��/�o�,1�t��m�h�o�������H�9�}��������h�ixսY��49X��\)�H�9�@���w�e`B�''���'��-�'H�9�'<j���P�e`B�H�9�8Q콃o�]/���-�aG��}󶽃o�y�#�q���]/��E���\)�����/��\)��{���B�gB�BB �B�iBqB&�TBo�B�QB	��B��BLBJB!B[BB�B&2�B.�B!PWBk�B��Bo+B�$B�B	crB%rB�B0�IB4�B'6yB
��BB7B{fB Q�Bh�B#�By�B ��B	˔B?�B��B$-�B!��B%B `B��B�B��B	�B%KB%�"B,��A�s)B $B�0Bg�B�%B��B)S�B��B�B :/B#�}B��BtqBLSB��BܜB
b~B	�B+�BЏB
oBoB�By�B?�B?�B.B E0B��B�B&��BC	BۦB	��B�*B<UB@yB<rB�Bl�B&�,B.*@B!5B�B�B>SB��B4�B	�|B$�B�[B0��B4[�B'=vB
�SB;,B<�B�6B �B�mB?�B�dB!7tB	��B��B�:B$@�B!ÌB�5B0�B�BAB/�B	˘B% �B%��B,�A��A���BdqBG"B7�BMkB)ktB�B�QB 5�B$�RB��BnBAWB�0B��B
F8BCHB?�B��B
��B@B6]B��@�bA�R�A3uAŝ�BxAn@�AA�!A�UA���B�+A�9�A��A�/6A�~�A��0A�^fAd�7AI;A��AAmt�AW��A��]@��,AZ%_Av>EA��AX:AO�@��LA|��Al�	@G
ZAz|A�CA�YA���>�K@㑓A�-%Aˌ�@���@2b^?��A� @�5�C��JC��VA�LA�lgA$�iA9Xe@��[A���A���AZ��A�dA��7A�)x@ɔ�A�]�A�x@��@�V
A���A�?�C���A�CC@�`A���C���C�*�A�O�A�HB�>�iA�@�SA��A3'AŎYB�Anב@�\A��`A�n}A��OB�bÄ́UA�]�A�PVA�{�A� A��@Ae AMA��Am�AU��A��x@�^,AZ��AujAׁ*AX�AO4@{VuA|�Ak�w@0,�Ay.PA�DA�k�A��0>ւj@�4BA�eA�t�@��`@3��?�@A�{�@��NC���C��A��A��[A"�A;A ��A��A���AZ��A�o�A��A�sN@ɒ�A��A�xT@�b@�^�A��"A�0C��NA�pN@�{?A���C��nC�$�A���A�(�B�>G\gA��               >                                    7                           	               #   	   F   (      %                        *      �   	      
   
   ,   	   -               $                  #   
            	      &         5   	                     G                           )         ?      )                                    !      A   !                              +      '               +      1               !            %      -                                                      +                                    ?      '                                          A                                 #                     %      1               !            %      /                                       N��mN�8CN�%�NVO�PXPOvKN�t�O*0�N+��N�PM���N���OF�0N���N��qN���P���Nv�O�l�N��<O�m2O=5O:�zN��MO��OkzQN
s�N�'HO8O@m3O�+�N��P��iO1�OB��Or�N�3�N��NMhN:.Oy#}O\�N��\O��OY�Or{PN4��N�jO
��OF�PO҇rN�wxP%�,N7'�OH�Nl�BNa2�O��3Nu�O(��Nc[�Oɫ8N�@O�_N�_O�N��N��ND2Nx�N���N�w7N��2O!N��N��TOH��  x  �  �  �  v  @  �  �  a  �  O  �  �  �  5  d  e  �  $    �    =  �  K  �  �      h  �  �  o  �  ]  �  D  �  ?  h  g  U  �  _  �  �  �  4    �  m  �  >  g  �  �  T  �    o  [  -  t  l  �  �  �  v  �  +  �  {  �  �  7  �  �<D��;��
;D��:�o���ͼo�D���#�
�D�����
��`B��`B�49X���ͼ#�
�#�
�D���49X�T���T����C���t�����ě��ě����
��1��j���ͼ�j���ě����ͽ,1��`B�+��������������/��h����`B�t��C���9X���#�
�o�o�#�
�+�C��t��t���P�#�
�#�
��w��w�''49X�<j�<j�<j�D���D���P�`�T������]/�u���-�}󶽅��� ���������������������������������������������������������������������������������������}�LO[hktvyxtph[YOJHILL:<IU[bcdb_UIF<<7::::��������������������Zaanz��znaZZZZZZZZZZggost����ytggggggggg������������������������

���������')5BEJNPUQB5)#�����������������������

���������mnz}����zynjkmmmmmm���0<b{�����{U<��������������������������������������

��������������

������������������������

��������ABGOQ[^himhc\[ZOGBAALNV[gtz�����tga[NLLL���������
 ("
����#/2<<</#"(**/56CO[YSOJC6*((((w����������������vuw<=IUbossrnibUKIB;=?<ft�����������tlg][^f��������������������wz�����������������w����������������~|mqz�����������zvmihm39BN[gnqqpnig[NB51/3��
#%+/#
�����uz�����zxuuuuuuuuuu��������������������fgnt����tttgffffffff�����������������/6EORTSOB3)��
"
����������������������������55BN[hmnga[NB=5/+).5������
���������	
�����������������������������������������������������T[t�������ztjgf^[XQT��
#0<V\^UI<, 
���!#&.06<>@@?<0,)&#! !����������������TT_abcdca\TTRRTTTTTTimoxz���������zumefi%/<EHJH?</)$%%%%%%%%#)--+)���
")+-&
���������

������������mov{������������{wnm�����������������������)5BOZkid[O6�������������������������� %-45/#
����������� ��������������������7<>HRUZ\YUH<;8777777������������������������������������}w
"###"
	!#(/6:940/#!  !!!!!!255BNQTSONIB=5422222ty�������������|wtst��������������������ABFOO[ahkhhb_[OGBAAA����������ۻ����������ûлܻ�ܻڻлû�������������¦£¦²¿������������¿²¦¦¦¦¦¦����������������,�4�=�4�(�����U�Q�H�G�G�H�U�a�f�j�a�a�U�U�U�U�U�U�U�U�������������������$�.�6�6�0�������ٿm�k�e�i�m�w�y���������������������y�m�m�����������ȼʼּ޼����׼ּʼ�������������žŽ����������������� ���������U�I�H�D�E�H�U�Z�Z�W�U�U�U�U�U�U�U�U�U�U���������������������������������������ǈǇ�{�q�{ǈǔǖǟǔǈǈǈǈǈǈǈǈǈǈàÝàêìï÷ùü����������ùìàààà�m�f�a�`�\�^�a�m���������������������z�m�z�x�t�z���������������������z�z�z�z�z�z�����������������������������������������/�.�%�/�7�;�H�R�T�U�T�K�H�;�/�/�/�/�/�/���������J�?�9�@�Z����������������������G�=�;�:�;�<�G�K�T�V�T�P�G�G�G�G�G�G�G�G�������s�b�M�Y�q�~���������Ѿվ׾Ҿʾ���ŠŔŖŞŠŭŮŹ��������żŹűŭŠŠŠŠ���y�m�`�R�M�T�`�m�y������������������������׾˾Ѿ׾����	������	���������������*�*�-�8�<�>�:�6�,�*����������}������������������������������������������	�������
�	�����Ŀ������������������������Ŀ̿̿οҿɿ��6�+�*�4�6�B�B�C�C�B�B�7�6�6�6�6�6�6�6�6�	��������ؾؾ�����	����	�	�	�	���������������������ʾϾ׾ھ޾޾ݾ׾ʾ��:�-�$�%�&�-�:�F�S�W�_�c�j�x�{�x�_�S�F�:�Ŀ��������Ŀѿݿ����������������ݿѿĿ`�W�W�`�c�m�y���������������y�m�`�`�`�`�ɺ������o�k�����ɺ��:�S�`�S�F�!���ɿ����������������Ŀѿݿ������ݿѿĿ����������������������
�����
��������)�����������)�5�B�D�N�P�M�F�B�5�)���������������������������������������׹ù��¹ùϹܹ�ܹԹϹùùùùùùùùùür�p�f�c�_�f�r�{�����������r�r�r�r�r�rŇŁ�{�u�u�{ŇŉŔŔŔŋŇŇŇŇŇŇŇŇàÙÓÍËÇ�z�u�x�zÇÓàð������ùìà�F�A�<�F�I�S�_�l�x�����������z�x�l�_�S�F�������������ɺֺ޺����ֺɺ����������@�3�'����$�'�@�L�~������������e�L�@���������*�6�?�C�D�H�F�C�>�6�*������������������'�0�:�;�3�'���F=F:F1F=FJFVF`FVFUFJF=F=F=F=F=F=F=F=F=F=FFFFF#F$F1F8F=FGF=F:F1F$FFFFFF�"���	���	��"�+�/�;�=�H�K�P�H�;�/�"������)�6�B�O�W�[�c�[�V�O�B�=�6�)����������������������������н����ݽн�������(�1�4�A�M�Z�a�b�Z�M�A�5�(���n�g�s���������������
��ּʼ�����n�g�f�Z�R�Z�g�s�������������s�g�g�g�g�g�g�(�����	�
��� �(�2�5�:�?�A�H�A�5�(����������	������	�����������������O�I�O�Q�[�h�t�x�w�t�h�[�O�O�O�O�O�O�O�O�5�!��$�)�B�N�g�t¡ �t�N�B�5²§§²µ¿������¿²²²²²²²²²²�'�����������'�4�@�D�J�M�M�B�@�4�'�g�b�^�g�l�s�z���������s�g�g�g�g�g�g�g�g���|�w�v�x�����������������������������������������������������������������������x�e�\�l�������ûлܻ���лû��������xìåäìñù��������������ùìììììì�������������� ����)�6�7�5�)����FFE�E�E�FFFF$F+F.F'F$FFFFFFF�������������������������������������������������������ʼӼʼ¼�����������������ĿĿĿ��������������ĿĿĿĿĿĿĿĿĿĿEEEE E*E7E>ECEJEPEWEQEPECE7E*EEEEE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��U�H�H�A�F�H�U�a�n�r�zÀ�z�r�n�a�U�U�U�U�<�/�%�)�/�<�@�H�U�\�n�z�~�z�v�n�a�U�H�<�����	�����������������������������Ϲ˹ù����������ùϹѹܹ�����߹ܹϹϽ.�!���������!�.�8�:�G�R�X�S�G�:�. R H B ` a  2 G y ` } I 6 A h 8 T ` 6 \ - T e  F L | H G 8 4 0 R   - 3 H b Z C G R ( / .  = @ L 2 P } N { E 1 7 3 < ) - u 7 L 6 O  : j B K c ? P R 2 [    �  �  1  �  .  "  �  n  �  C    �  �  �  �  �  =  L  �  �  y  ,  �  �  L    }    l  �  v    �  x  �  �  �  .  n  H    �  �    �  �  H  �  :  �    &    �  i  �  y  �  �  i  p  �  �  q  �  e  �  �  �  /  �  �  �  m  7  �  �  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  ?C  x  p  e  W  D  ,    �  �  �  �    Z  <  (     -  A  B    �  �  �  �  �  �  �  �  �  �  �  �  W  (  �  �  �  J    �  �  �  �  }  p  _  L  7      �  �  �  �  h  @    �  u    �  �  �  �  �  �  a  7  	  �  �  r  9    �  �  P    �  �  =  F  4  "     �  �    v  d  B    �  Y  �  H  �  V  �  �  �  
    +  2  7  <  ?  @  ?  :  0      �  �  �  T     �  T  �  �  �  �  �  �  �  �  �  �  �  ^  %  �  �  K  �  >  �      2  R  i  |  �  �  u  _  A    �  �    L    �  �  �  a  d  g  j  l  m  n  p  q  s  t  u  w  r  a  P  =  !    �  �  �  �  �  �  �  |  u  n  g  ^  T  K  A  7  -  #        O  L  I  F  C  @  <  9  6  2  ,  "        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  u  |  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  a  E  !  �  �  �  �  G  �  �  @  �  �      #  /  >  _  ~  �  �  �  �  �  �  E     �      5  3  0  .  )        �  �  �  �  �  �  p  U  <  #  
   �  d  a  ]  Z  V  S  O  L  I  E  ?  5  +  "         �   �   �  d  I    �  �  4  �  �  ,  �  �  T  )     �  �  l  *  �   �  �  �  �  �  �  �  �    v  n  c  W  J  =  0       �   �   �    "      �  �  �  �  �  �  u  X  -    .    �  h  �  P    �  �  �  �  �  �  q  M  "  �  �  �  u  J     �  �  f    M  |  �  �  x  h  R  7    �  �  �  ?  �  �  G  �  }  	  �    �  �  �  �  �  �  ~  T  &  �  �    H    �  �  d  G  2  =  =  ;  9  )    �  �  �  �  e  2    �  �  V    �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  C    �  �  s  *  �  ]    -  ;  E  J  I  @  0    �  �  �  g  0  �  �  f  !  �    �  �  �  �  z  f  R  >  +      �  �  �  �  ~  N     �   �  �    |  y  g  Q  <  &    �  �  �  �  �  ~  e  K  1    �            �  �  �  �  �  �  �  k  P  3    �  �  �  c  �  �      �  �  �  �  �  �  u  T  0    �  �  s    �  F  h  \  L  9  '             �  �  �  �  �  �  �  -  �  o  �  �  �  �  �  �  �  �  �  y  Y  1    �  �  I  �  t  �  Y  �  �  �  �  �  �  �  �  �  t  b  N  8    �  �  �  x  9   �  o  ^  :    �  �  �  �  �  �  �    g  G    �  F  �  �  �    5  G  W  k  t  �  �  �  �  �  {  N    �  T  �      	  O  X  ]  [  U  L  ?  ,    �  �  �  �  f  >    �  i  �  �  |  �  �  �  �  �  �  �  ~  V  %  �  �  a  �  �    �    �  D  A  >  <  7  .  %        �  �  �  �  �  �  �  w  c  O  �  �  �  �  �  �  �  �  �  {  i  P  8      �  �  �  �  �  ?  8  1  )       �  �  �  �  �  �  {  \  <    �  �  �  �  h  ^  T  J  @  6  ,  "        �  �  �  �  �  �  �  �  �  I  b  c  M  .    �  �  �  �  �  a    �  �  F  �  j  �    O  S  U  T  M  =  %    �  �  �  �  w  F    �  �  D  ,  �  �  �  �  �  �  �  �  �  y  j  U  3    �  �  T    �  �  H  T  S  Y  ^  \  ^  \  T  N  G  >  /    �  �  =    �  A  �  �  �  �  �  �  �  �  �  �  �  �  s  T  2    �  �  ;  �  I  �  �  v    u  �  �  �  �  �  Q  �  ;  �  �     u  �  	�  �  �  �  �  l  V  ?  &      �  �  �  �  �  �  {  D  	  �  �  �  �        +  2  3  -    �  �  �  q  9    �  W  �  �          �  �  �  �  �  �  �  �  �  v  X  9    �  �  �  �  �  �  �  �  �  �  �  �  o  X  V  m  w  h  \  ^  ]  G  0    R  h  m  m  b  [  P  9    �  �  �  G  �  �  Z    �  �  �  �  w  h  X  P  ]  i  V  ?  '    �  �  �  �  �  a  7    =  /  
  �  �  �  e  ;    �  �  �  K     �  >  �  S  �  {  g  m  t  z    ~  }  |  {  y  w  u  n  c  Y  N  D  :  1  '  �  �  �  m  G  !  �  �  �  �  g  @    �  �  K  �  �  A   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  }  }  }  }  N  P  R  R  N  I  A  6  +        �  �  �  
  (  5  =  D  �  �  �  �  l  F    �  �  A  �  �  �  ;    �  �  !  �  #             �  �  �  �  �  �  z  J    �  �  t  `  F  -  o  n  n  j  e  _  X  N  D  8  *      �  �  �  �  �  b  6  [  Q  G  >  4  *           �  �  �  �  �  �  �  �  �  �  -  *      �  �  �  �  �  �  �  Z  .  �  �  �  R    �  C  g  n  r  k  ]  I  7  '      �  �  �  �  �  l  5  �  �  (  Z  h  k  `  H  +    �    *    �  �  �  r  H    �  �  `  �  �  �  �  q  \  F  0      �  �  �  u  >    �  �  I    �  �  �  �  �  �  �  �  �  �  }  k  W  B  ,       �  �  "  �  �  �  �  �  �  �  w  _  >    �  �  �  \  !  �  �  1  �  v  d  R  @  3  "    �  �  �  �  �  �  z  d  O  9    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _  3    �  �  +  (  %  "              	  �  �  �  �  �  �  �  �  �  �    ;  X  �  �  �  �  �  �  �  ]  .  �  �  _    �      {  v  o  g  ]  U  R  F  .  
  �  �  X    �  �  M    �  x  �  �  �  �  �  �  �  �  w  c  >    �  �  H    �  q  &  �  �  A  �  �  #  R  u  �  �  j  4  �  �  `  
  �  F  �  �    7       �  �  �  �  �  p  O  ,    �  ^  �  �  Q     �   �  �  �  �  �  �  �  �  �  �  r  X  ;    �  �  �  �  J  x  �  �  �  �  v  S  .  	  �  �  �  i  ?    �  �  1  �  �  �  �