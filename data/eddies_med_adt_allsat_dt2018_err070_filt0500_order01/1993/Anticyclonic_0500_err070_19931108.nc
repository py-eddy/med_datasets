CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�n��O�        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mʇ;   max       Pz*�        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       =�x�        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @F}p��
>     @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�fffff    max       @vhQ��     @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @Q@           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @��            8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��t�   max       >`A�        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��x   max       B2�        :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B3"        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�   max       C��        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?w   max       C�d        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mʇ;   max       P&�H        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�k��~($        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��/   max       =�        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @F}p��
>     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?׮z�H    max       @vh(�\     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q@           �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�@            \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?uL�_��   max       ?�hr� Ĝ     �  ^�      	      c   2   )      >            N         V                  %                  6   	      V                                              ;                              K                        �   	               	         EOV�N�/�OD/Pz*�PjP5
Nt�P�N]��NǀVN��O��N�aoNM�P'�OP��O=��OiuwO�FXN�^�O��O,iN;9�O�%�O7B�N0�ON�~Nº�N%��PL��N$�sOF�OҘN��TNr�N�6�O�:�O�e�N��N�p�N*�O28�N�XhMʇ;O���O3eO:�:N�֭N�)nO�p�O�,PNS�N�N��P&�gN�YN�� N=�+N��qN�$�O&��NX �P�]N/qINM��NFO;�N_M��O0(�N�eOF+���`B�#�
��o%   ;D��;��
;�`B;�`B<t�<#�
<#�
<49X<T��<T��<e`B<�o<�C�<�t�<�t�<�t�<���<���<���<��
<��
<��
<��
<�1<�9X<�j<ě�<���<���<���<���<�/<�`B<�<��=\)=\)=\)=�w=#�
='�=0 �=0 �=0 �=49X=D��=H�9=P�`=P�`=T��=]/=aG�=e`B=m�h=y�#=��=��=�7L=�hs=��
=��=��=���=�1=�-=�Q�=Ƨ�=�x�$017<@D@<60,#==?BOY[bhkh[OB======����������������������������

�������`fnz�����������zria`���)B[t|mfn_B)���������������������&(5>BONh�������thB6&KU\anottna\UKKKKKKKK		 "/;<A;;/$"			A=?HTahga`THAAAAAAAAc`aht������������pgc����������������)5BCB?5)��������������������������"!�����=>LN[gtwyyvtlg[VLDB=���	!).11/,)�$%5>INV[gmt}��tg[B)$-/-/<BGHKKJH<71/----�������
 
 ����#/<>EKHE</#%&***%#)5CN[gt{{xtfgRNA=/#���������������������������������������������������������������������������������������������������6O[hlmhOB6*���������������������������������������������������������������������������������������������������������


�������#),5<HUn���znUH</tnioq�������������{t��������������������


#%,-,#




����������������������������������������KEN[dgltzytmg[XNKKKK<2<IUVURI<<<<<<<<<<<tpsz���������������t�
#0<GIKG<0#
�� 
#/23463/#
&'%()/<=<=<72/&&&&&&,+6BDO[`hljh[UOHB6,,����)BSVOC,���/5BNSTSNB5)��������uihbhu���������������������������������������������������)6KTYOJB6)����������������������sxz�������������}zss��������������������
*,6@BCCC64* �������)36;=>=5)#��������������������c]`gt�������������nc;85<@HJNPHA<;;;;;;;;)220)559?BNQQNFB555555555)!)068;;:60)��������������������/#"#/12///////////lebfnz����������{znl �
#$*)$#
������	���������˻x�����������Ļû��������������{�x�n�m�x�ܻ�������������ܻۻٻԻлܻܻܻܻܻ�����������������������ƿƿƻ�����������پ�4��������q�m�o�Z�A���սн߽����������������������������}�N�D�B�J�Z�������m���������������������m�T�G�;�3�4�>�M�m���������������������������������������Ѽf������ļȼּ׼ż�������r�f�]�Z�U�V�f�����������������}�����������������������a�a�g�m�n�r�o�m�a�U�T�P�H�G�H�H�T�a�a�a�/�;�H�R�S�T�H�;�/�/�%�'�/�/�/�/�/�/�/�/���
��#�+�3�5�:�/��
� �����������������a�m�s�t�m�l�i�i�d�a�T�T�P�J�J�M�T�_�a�a����������������y�{����������������������4�M�f�������f�@�'���ܻû��ǻ�����#�*�6�B�B�@�6�*�������������uƁƎƏƐƑƉƁ�u�h�\�W�O�K�M�O�\�h�r�uƧƳ��������������������ƳƧƚƔƄƋƛƧ��$�?�I�b�m�b�`�L�=�/�$������ �������������������������������������������������������������������������������������!��������������������������m�y�������������������y�m�k�m�m�m�m�m�m�a�h�n�z���������m�a�T�I�;�/�"�!�"�/�H�a�������������	�����������������������������ʾʾʾɾ���������������������������Óàìù����������ùìàÓÇÆÀÁÇÉÓ�F�:�:�9�-�!��������!�!�-�:�D�F�F�t��z�w�t�q�h�[�V�[�h�m�t�t�t�t�t�t�t�t���׾���������׾�p�Z�U�Z�f���������H�T�[�a�c�a�T�H�B�=�H�H�H�H�H�H�H�H�H�H�s���������������������s�q�l�j�h�n�s�s�<�H�U�_�_�Y�U�L�H�<�/�#��"�#�.�/�8�<�<�s�t�y�s�n�f�Z�M�F�A�9�A�M�Z�f�l�s�s�s�s�����ʼʼ̼˼ʼȼ������������������������/�1�<�A�A�=�<�2�/�)�#������#�'�/�/���&�5�A�N�V�W�V�L�I�A�5�'�$�"�����ѿݿ�����
�	�����ݿѿĿ��������ĿѼ@�M�R�N�M�N�M�B�@�7�4�2�4�8�<�?�@�@�@�@�`�m�y�y�y�y�p�m�`�T�G�@�G�P�T�`�`�`�`�`àìëìöìàÓÏÓÕÜàààààààà��������������������������������úúý�Žнݽ������ݽнɽĽ��½Ľͽннн���� ������������������6�O�[�k�{���t�[�B�6�)������������6�����������������ּɼʼ̼Ӽּ���񾘾��������Ǿʾ־Ѿ��������������������������	�����	�������ܾ��������#�)�.�/�;�6�/�)�#����
��
�����;�H�T�a�����������z�m�a�T�H�;�/���"�;��#�?�I�U�`�`�M�<�0�#�����������	��������������	�����������������������������������������������������������������������������~�������������S�x�������|�s�_�S�:�!�����������!�S�ܹ��������������ܹӹԹԹڹܹ�ÇÓÕàæêìôìàÓÍÇ�{�z�u�z�~ÇÇ�\�e�h�s�h�b�\�U�O�G�O�[�\�\�\�\�\�\�\�\�S�`�l�n�y�z���y�u�l�`�W�S�R�G�G�F�G�G�S¿����������������¿²¦¡£¦²³¼¿¿�N�[�g�t�t�g�[�N�B�5�*�0�5�B�N������ ������ֹܹܹܹ�����������6�B�L�T�`�b�^�W�O�6����������	��ECEPE\E_E\ETEPECE7E,E7E?ECECECECECECECEC��(�2�5�=�5�(��������������N�O�Z�g�i�i�g�Z�V�N�I�I�N�N�N�N�N�N�N�N�@�L�R�Y�e�~�������������������~�r�_�L�@�������������ּּԼּ�������ĳıĳĿ��������Ŀĳĳĳĳĳĳĳĳĳĳĳ�ܻ����
����������ܻлûȻллڻ��{ǈǔǡǭǪǢǡǔǈ�{�z�p�r�{�{�{�{�{�{E�E�E�E�E�E�E�E�E�E�EzEuEtEuE�E�E�E�E�E� E O 7 2 ( < [ 0 b N ! 6 r a N B   = E j 8 C f 1 I G  P T , 8 Q R = t ; k  � Q U 9 B ~ J X 5 n S i \ h N B * , G ; 8 P i  5 z : Y \ Q ? E  /  T  �  _    �  @  �  �  �    �  y  s  �  J  �  �  �  �  �  �  \  |  �  �  V  �  �  T  �  4  u  ?  �  �  �  �  1    �  X  �  �  8  |  �  �  �  +  (  �  �  M  �  �  	    ;      �  b  V  B  _  O  �  ~    s  �  ���t���o<49X=ȴ9=T��=<j<D��=�O�<D��<�C�<�1=�9X<�1<�t�=���=o=C�=t�=t�<���=aG�=�P<�j=t�=C�<�`B=��P<�h<���=�/<�`B=��=�P=�P<�h=�P=#�
=T��=�P=8Q�=@�=�+=49X=49X=���=y�#=�hs=D��=P�`=�hs=��-=aG�=aG�=]/>o=�hs=�O�=�hs=��w=��w=��
=��T>`A�=�9X=�-=�1=�
==��=\=���=��#>:^5B%t}B�Br�B"^�B �fBĵB��B��B+�A��xA�crB
�XB�lB�B!gCB�B	�B�Bs�B��B:�B�B/, B�TB��Ba�B"%�B�RB(�BK�B��B��B��B��B"�}B3AB��B
��BkBt�BѻB�fB��B&�BtB%��B�oB4�B�LBPgB�hB2�B*��B,üB�@B(B ��BGB/�oB��B�.B!U)B�B"�B�}B�B
B��B�B�5B#�Bf>B%�	B@^BV�B"@B RB�9B8�B<hB�LA���A�wB
��BGgB=sB!��B;�B		BFB~nB��BC`BT�B/:&B=�B�B�IB"@ BI
BC>B>�B�TB0�B�B��B"��B�B��B�B��BCtB�lB�JB	�B&�RB ��B%H�B��B<�B*B\yBA�B3"B*�B,�BIB=�B }cB?�B/��BIB�_B!AB=�B�B��B��BB�B�8B2�B@B<B��@��@�HB��A6dA��
Ak.A�U�@��A�HJA�&$A��pA��A��.AFؖ@�:�A��[B�BwPB

�A�_�A�|�A�T
Ao,�A���A��qAN�oA�R@n�A��uALA	A�5�AD��AýOA?Lm@�>|A�XA�ɈA}�(@��AiA˟�A��WA)��A��2A��A��AKj�AY"yA�8>A�+A�@�AY��AA��@�|?��A�e�B̆AA��JA��c?�A�H�C��iA�V�A�b�@ �1A��A�p@��BMC��@��s@���B�eA6�iA�SuAi*A��	@�A��PA�2oA�~:A�d�A�~�AE	�@�NA��jBr�B��B	�A�|OA�w�Aҫ)Ao�VA��A�v[AN��A�}�@dqHA� DALA���AE �AĂ�A@��@�A���A��jA}�@� eAi-�Aˀ A��A)�VA���A�t�AKAJ�WAYلA���A���A�y�AZ�A��A��@� ?.O�A�m4B�iA��A�~�A���?wA�5C��oA�A�gc@ ��A �A�8@��RB��C�d      
      c   2   *      ?            N   	      V               	   %               	   7   	      V                                              <                              K                        �   	               	          E            7   1   -      -                     1            #               !                  /                                             '               )               )                        %                                          +   !                                       #                                                                                                            '                                                   N�AdN�/�O�fO�6)P&�HO�DNt�O��N]��N�QNO[rO:MN�"NM�OJ�jOP��N�)O�KO�FXN]/�OS�8O,iN;9�O�NэN0�N�0:Nº�N%��O�x�N$�sOF�OҘN��TNr�N�6�O�:�O��N��N�p�N*�N�U�N�XhMʇ;O�RO5O�N�֭N�)nN�+�O��eNS�N�N��P:N�vN��N=�+N�tVN�$�O&��NX �O�5&N/qINM��NFO;�N#�M��O0(�N�eOBR�  �  3  X  �  }  �  �  ;    @  s  �    �  �  �  e  �  �  �  v  �  H  r  �  �  �  w  �    �  �  �  �    �  \  x  �  m  �    �  (  �    m  A  #    [  '  4  }  	  �  �  C  �  R  i  �  �  �  h  �  �  d  �  o  �  7��/�#�
%   =,1<T��<u;�`B<���<t�<49X<e`B=#�
<e`B<T��=ix�<�o<�9X<ě�<�t�<��
<�<���<���<�j<ě�<��
=#�
<�1<�9X=}�<ě�<���<���<���<���<�/<�`B=o<��=\)=\)=8Q�=�w=#�
=]/=49X=@�=0 �=49X=m�h=P�`=P�`=P�`=T��=�+=e`B=q��=m�h=}�=��=��=�7L=�=��
=��=��=���=� �=�-=�Q�=Ƨ�=�#&06<=B?<50*#==?BOY[bhkh[OB======�������������������������������������gfkqz������������xpg ��)5BNYZUVNB5 ��������������������:68<IQ[ht������thOB:KU\anottna\UKKKKKKKK"#/9;>;5/"FBHJTabcaZTHFFFFFFFFsmoty������������tss�����������������)5BCB?5)��������������������������"!�����KINP[gstvvtrg[SNKKKK	 '),*)&$%5>INV[gmt}��tg[B)$:4035<=CHIIHD<::::::��������

����#/<>EKHE</#%&***%-8=BGN[gtxwtk_]NC<5-���������������������������������������������������������������������������������������������������'').6BOQZ^_``]XOB6,'���������������������������������������������������������������������������������������������������������


�������#),5<HUn���znUH</plt�������������|up��������������������


#%,-,#




����������������������������������������KEN[dgltzytmg[XNKKKK<2<IUVURI<<<<<<<<<<<wux��������������{w
#00<EIJIF<0#
#*/2342/#
&'%()/<=<=<72/&&&&&&,+6BDO[`hljh[UOHB6,,)6BFNEB6+) ��)5BNRSRNB5) ������uihbhu���������������������������������������������������)BNTOHC7)�����������������������yz|������������zyyyy��������������������*6?AB6.*&�������)36;=>=5)#��������������������sljpt�������������ts;85<@HJNPHA<;;;;;;;;)220)559?BNQQNFB555555555)!)068;;:60)��������������������/#"#/12///////////lebfnz����������{znl �
#$*)$#
��������	�������ʻx�����������������������������}�x�q�x�x�ܻ�������������ܻۻٻԻлܻܻܻܻܻ���������������������������ƽ�����������̾(�A�M�U�Z�V�M�A�8�������������(���������������������������s�Z�L�L�Z�����`�m����������������y�m�T�G�?�:�9�@�T�`���������������������������������������Ѽf�r����������Ǽ¼���������r�j�d�a�b�f�����������������}�����������������������T�a�f�m�m�q�m�h�a�[�T�R�H�I�T�T�T�T�T�T�/�;�F�H�L�J�H�;�7�/�*�-�/�/�/�/�/�/�/�/���
���#�(�,�,�#����
���������������a�m�q�s�m�j�g�f�a�a�]�T�S�L�L�N�T�`�a�a����������������y�{����������������������'�4�:�@�B�B�@�4�'����������������#�*�6�B�B�@�6�*�������������h�u�ƁƈƊƁƀ�u�h�`�\�P�R�\�d�h�h�h�hƧƳ������������������ƳƯƧƚƘƗƚƦƧ��$�?�I�b�m�b�`�L�=�/�$������ ���������������������������������������������������������������������������������������!��������������������������m�y�������������������y�m�k�m�m�m�m�m�m�T�a�m�r�z�����x�m�a�R�H�;�/�&�/�;�C�T���������������������������������������׾����ʾʾʾɾ���������������������������Óàìù��������ù÷ìàÓÒËÓÓÓÓÓ�F�:�:�9�-�!��������!�!�-�:�D�F�F�t��z�w�t�q�h�[�V�[�h�m�t�t�t�t�t�t�t�t�����ʾ׾����׾ʾ���������u�p�w�����H�T�[�a�c�a�T�H�B�=�H�H�H�H�H�H�H�H�H�H�s���������������������s�q�l�j�h�n�s�s�<�H�U�_�_�Y�U�L�H�<�/�#��"�#�.�/�8�<�<�s�t�y�s�n�f�Z�M�F�A�9�A�M�Z�f�l�s�s�s�s�����ʼʼ̼˼ʼȼ������������������������/�1�<�A�A�=�<�2�/�)�#������#�'�/�/���&�5�A�N�V�W�V�L�I�A�5�'�$�"�����ݿ�������������ݿѿĿ������Ŀѿݼ@�M�R�N�M�N�M�B�@�7�4�2�4�8�<�?�@�@�@�@�`�m�y�y�y�y�p�m�`�T�G�@�G�P�T�`�`�`�`�`àìëìöìàÓÏÓÕÜàààààààà���������������������������������������ҽнݽ������ݽнɽĽ��½Ľͽннн���� ������������������B�O�[�h�q�t�r�f�[�?�6�)������!�6�B�ּ�����������������޼ּ̼˼̼Լּ־����������þʾоʾ��������������������������	�����	�������ܾ��������#�)�.�/�;�6�/�)�#����
��
�����m�z�|�������������z�m�j�d�a�`�\�a�b�m�m�
��#�=�D�N�X�\�J�<�0�#�������
�
��	��������������	�����������������������������������������������������������������������������~�������������S�_�x�����w�u�`�S�:�-�!���� ���!�:�S�ܹ�������������ܹٹֹչ۹ܹܹܹ�ÇÈÓàäçéàÔÓÇÁ�z�y�zÆÇÇÇÇ�\�e�h�s�h�b�\�U�O�G�O�[�\�\�\�\�\�\�\�\�S�`�l�l�x�y�t�l�`�S�I�G�F�G�H�R�S�S�S�S¿����������������¿²¦¡£¦²³¼¿¿�N�[�g�t�t�g�[�N�B�5�*�0�5�B�N������ ������ֹܹܹܹ�����������)�6�B�F�O�T�V�V�O�B�6�)�������ECEPE\E_E\ETEPECE7E,E7E?ECECECECECECECEC��(�2�5�=�5�(��������������N�O�Z�g�i�i�g�Z�V�N�I�I�N�N�N�N�N�N�N�N�@�L�R�Y�e�~�������������������~�r�_�L�@���������ؼּռּ����������ĳıĳĿ��������Ŀĳĳĳĳĳĳĳĳĳĳĳ�ܻ����
����������ܻлûȻллڻ��{ǈǔǡǭǪǢǡǔǈ�{�z�p�r�{�{�{�{�{�{E�E�E�E�E�E�E�E�E�E�E�E�E{EuEtEuE�E�E�E� A O 9 & 3 < [ 5 b L 2 # o a / B  2 E \ $ C f 1 C G & P T > 8 Q R = t ; k  � Q U B B ~ ; E - n S P Z h N B + - B ; : P i  4 z : Y \ > ? E  0  "  �  /  �  �    �  �  �  �  g  M  F  �  �  �  �     �  �  �  \  |  �  �  V  �  �  T  [  4  u  ?  �  �  �  �  �    �  X  �  �  8  �  6  K  �  +  9  a  �  M  �  N  �  �  ;      �  b    B  _  O  �  H    s  �  �  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  x    �  �  �  z  s  j  `  R  B  0      �  �  �  �  �  �  3  !      #           �  �  �  �  r  ,  �  �  -  �  {  W  X  V  R  M  G  @  >  =  0       �  �  �  �  R    �  M  u     o  �    b  �  �  �  �  �  �  e  3  �  ]  �  �  p  U  )  J  f  x  }  w  g  V  L  >  @  M  8  �  �  2  �  8  �    �  �  �  �  �  �  �  �  �  �  �  �  f  <  �  �  4  �  F  �  �  �  �  �  �  �  �    n  X  A  *      �  �  �      &  ,  �  �  �    -  :  0    �  �  v  L    �  T  �  �  K  z        �  �  �  �  �  �  �  �  r  `  S  S  R  R  Q  P  P  3  8  <  ?  =  <  5  *         �  �  �  �  �  a  1  �  �  K  T  ]  d  k  q  r  r  n  f  Z  ;    �  �  e    �  m    	�  
  
g  
�    v  �  �  �  �  q  @  
�  
�  
  	4  @  �  �  P  	  
      �  �  �  �  �  �  �  �  �  �  �  s  C    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  a  Q  B    g    n  �  .  c  �  �  �  �  �  V  	  �    �    r  �  �  �  �  �  �  �  �  �  �  �  o  V  >  "  �  �  �  q  P  :    2  G  V  a  e  d  _  V  F  ,    �  �  ~  J    �  �  P  t  �  �  �  �  �  �  �  �  �  �  �  q  M  %  �  �  �  j    �  �  �  �  �  �  �  �  o  R  5    �  �  �  y  6  �  �  ?  �  �  �  �  �  �  �  �  �  s  N  "  �  �  �  P     �   �   6  4  `  k  q  t  u  v  n  X  ,  �  �  �  _  .  �  �  i    �  �  �  �  �  �  n  Q  0    �  �  �  W    �  p    �  S  K  H  A  ;  4  .  '             �  �  �  �  �  �  �  �  �  ]  [  _  j  q  m  d  U  H  @  (  �  �  �  �  c  A  )  *  *  �  �  �  �  �  �  �  �  �  r  W  9    �  �  �  t  6    �  �  �  {  `  F  ,    �  �  �  �  �  �  �  ~  }    �  �  �  �  �    :  X  j  z  �  x  N    �  \  �  K  �  �  �  F   �  w  m  c  Y  O  F  ?  7  /  &      
  �  �  �  p  A    �  �  �  |  q  e  Z  O  @  /      �  �  �  �  �  �  �  �  �  .  �  -  �  �  �  �        �  �  �  8  �  `  �    3  �  �  �  �  �  �  �  �  �  �  �  �  �  |  q  e  Q  9  !  
  �  �  �  �  �  �  �  �  �  w  ^  @     �  �  �  Y    l   �   �  �  z  l  Z  F  0    �  �  �  �  j  O  E  9      �  �  �  �  �  �  �  �  �  �  �  {  [  3    �  �  =    �  �  �  Q        �  �  �  �  �  �  �  �  �  �  u  W  8    �  �  �  �  �  n  O  0    �  �  �  �  |  d  N  7    �  �  �  |  ^  \  L  ?  7  '      �  �  �  �  �  p  ]  G  2      �  �  W  l  x  r  d  T  @  -       �  �  �  ~  I    �  x    �  �  r  e  W  ?  %    �  �  �  w  V  6      	  �  �  �  �  m  Y  E  5  %      �  �  �  �  �  �  �  n  M     �  �  �  �  �  �  �  �  �  �  �  �  �  .  �  _     �  �  U    �  x  �    5  P  g  x  ~  v  g  D    �  �    H    �  �  w  O  �  �  s  a  N  6      �  �  �  �  �  �    l  N  .     �  (  .  4  :  ?  E  J  P  U  [  t  �  �  �  (  8  >  E  L  R    p  �  �  �  �  �  �  {  L    �  e  �  R  �  �  d  �  �  
          �  �  �  �  �  �  u  V  .  �  �  L  �  k   �  Z  f  l  l  e  W  A  $  �  �  �  M    �  {  2  �  �  @    A  -      �  �  �  �  �  v  [  @  !     �  �  �  u  Q  -  #              �  �  �  �  �  �  �  �  �  �  �  |  g  �     �  �  �  �  �  �      �  �  �  �  w  .  �  7  �   �  M  Y  X  H  3    �  �  �  �  �  U  '  �  �  �  8  �  "  ~  '         �  �  �  �  �  �  �  �  �  w  i  ^  S  I  >  4  4  ,  %             �  �  �  �  �  �  �  �  �  �  �  �  }  {  y  w  u  s  q  o  m  k  k  m  p  r  t  w  y  {  ~  �  �  �  	  	  	  �  �  �  �  �  a    �  j  �  a  �  �  �  "  �  �  �  �  �  �  �  {  Y  4    �  �  y  @    �  J  �   �  �  �  �  �  �  �  �  �  �  s  U  4    �  �  �  k  6  �  �  C  �  �  �  F    �  �  >  �  �  m  #  �  �  =  �  q    �  h  �  v  e  S  :    
  �  �  �  �  p  :  �  �  e     `   4  R  @  &    �  �  �  j  B    �  �  �  K    �  �  �  i     i  ]  I  1    �  �  �  �  v  `  H  (    �  �  o  3  �  �  �  �  t  \  >  &    �  �  �  �  g  ?    �  �  r  (  �  �  �  �    m  �  �  �  �  �  �  �     �  �  �  �  I  	�  �  B  �  �  �  f  J  -    �  �  �  {  P  ,  6  @    �  �  s  :  h  Z  L  >  0  !      �  �  �  �  �  �  �  �  i  H  '    �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  q  g  ^  U  L  �  �  �  �  �  �  j  K  +    �  �  g  ;  �  �  1  �  6  �  I  I  K  Y  d  e  b  W  F  -    �  �  �  d  8    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  [  F  ;  5  :  ?  o  ?  	  �  �  H    �  �  A  �  �  O    �  u     �  +  �  �  �  �  �  c  @    �  �  �  [    �  �  <  �  �  #  �  V      �  �    L    �  �  G  �  �  (  w  �  
~  	f  C  
  