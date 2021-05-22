CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�z�G�{       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N=P   max       P�+e       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��v�   max       =C�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�\   max       @F\(�       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vmG�z�       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P@           �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @�Ԁ           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �I�   max       <���       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��_   max       B,O�       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B-?T       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >� �   max       C�0�       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C�*$       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Q       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N=P   max       P�J       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?���Fs��       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �\   max       =C�       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?^�Q�   max       @E�p��
>       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vm�Q�       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @R            �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @�o�           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?wX�e+�   max       ?�4�J�     0  ^            "            &            G               $   	      	   P                     A      ?                              !      	                           &                     $                     	         	         -Np@O_�Na~�O��tO	��O?<�OW<P/+.NNXO22�N���P�+eON$KN��NC�ZNI��O��O6=�O�&N��P��YN���O�0OEIJN���O��O��}PmTN&�PQN�OaRN��O��N�C�O_�Ov#�N�C�N�I'O`}O�@�O"��N���N��aN��OLH[N4]�Ot�Ol~,N<o$N?c�O��N��O"~�PY�N6z+N���O9!�O��SO��N�u�O �fO�2�O&�\NkИO&��N6��Nt`cN=PN�L�N���OZ�=C�=C�<�j<D��<#�
<t�;�o;�o;�o;o:�o:�o��o�o�D�����
�ě��#�
�#�
�49X�49X�D���D���u��C���t���t����㼛�㼴9X��j�ě��ě����ͼ��ͼ��ͼ���������/��`B��`B��`B��h���o�o�\)��w��w�#�
�0 Ž0 Ž49X�49X�<j�H�9�P�`�]/�]/�e`B�e`B�e`B�m�h�u�u������1��{��{��vɽ�v�5<HMKHF=<40155555555nz|����������zwonnnn����
��������������������������������DHOU]anz}{zngaUHFB>D�� �����ht�������������wtokh��������'"��������Z[hkt�trh[ZUZZZZZZZZ	
#*/9<CD></#	����������������������!0U������nbI<�����
#'/50/#
���������������������������������������������KO[hqqh[QOKKKKKKKKKK��������������������HMT^jmqxrmjfca^THEEH����� ������������

 #).#








6U�����7A�����bI36����������)5=BEHHB5)GTalmppifbaTSMHFB@AG������������������������������������������������������������)B[gbgfk~}o[B95<9/&) #/<@<:/#"          ����������������������������������������ttu�����������tjkmtt�������
	���������)+.,,))")5:;:5)&��������������������BHU\anpznkaUSHG@BBBB�������������������������
#,# 
��������dgt�������������m]^d��������

��������?HJUafeaa[UJH<??????������������������������������������������������������������#/<>=</#imtz�������zumjfefiiNP[gt��������tgd[VPN��������������������IO[]]\[YOLIIIIIIIIII���7B=6)"������DOT[hotz�~yth[OMDDDD��������������������Z^n}����������tgYRWZ��������������������������������������������������������������������������������������������������SUUaknz����zsna[UQSSzz�������������zvvzz����������������~zz��	#&'&# ��#+//-#*037<IMPTQNII=@0,+'*�������������������������������������������������������������������������


	#&/3:=8/#		ĿĶĵĿ����������������ĿĿĿĿĿĿĿĿ���������������ʼּ����ټּʼ��������a�Y�U�M�N�U�]�a�n�x�y�n�a�a�a�a�a�a�a�a��������������������%�3�:�?�/��
������������������������������������������������������������$�0�=�I�S�O�I�=�0�$���ѿοοѿֿݿ��������������ݿ��������m�h�������������������������������e�b�b�e�n�r�|�~�����~�r�e�e�e�e�e�e�e�e���������������������������������������H�E�;�2�;�=�H�T�a�m�q�o�m�a�U�T�H�H�H�H�������N�@�;�*�N�g�����������������׾4�4�/�3�=�A�M�Z�f�g�p�s�u�w�s�q�f�Z�A�4àÜÛÜàáìîù��������üùìàààà�H�H�D�H�L�R�U�a�e�a�\�Z�X�U�H�H�H�H�H�H�T�Q�K�M�T�a�k�j�a�]�T�T�T�T�T�T�T�T�T�T�Y�M�(� �,�4�@�Y�_�f������������������Y����ƭƧƚƘƚƳ��������������� �������ʻû����������ûлܻ�����ܻлûûûü��������������żǼż����������������������������}�����ɽ�!�<�W�S�!����ۼؼʼ��������z�|�������������������������������N�J�C�F�N�T�Z�g�s�~�~�z�}�s�i�g�a�Z�N�N������(�5�N�Z�g�p�m�g�Z�R�N�A�5�(��L�J�F�@�@�9�@�L�Y�e�h�r�z�~�r�e�[�Y�L�L������������ƺƵ������������#�%��������������������������*�6�C�C�6�/�������������������B�O�[�d�e�[�P���������H�B�@�@�H�Q�U�Z�V�U�H�H�H�H�H�H�H�H�H�H�ƺ��������ֺ��!�:�S�g�b�X�F�-�!�������	����������#�)�+�1�4�)����������������������������������������������`�T�G�>�3�1�0�6�;�G�T�^�g�����������y�`�����������������������������������������g�h�n�t�yćĚĦĳĶĵĳĪĦĚčā�x�h�g�;�:�>�A�F�F�T�a�g�m�r�w�z�~�z�m�a�T�H�;�������������������������������������������������������������������������������������������������������ŽǽĽ����������m�i�^�W�W�T�V�a�z���������������������m����������������������������������������������������������������������������������������*�/�6�6�?�6�*�������Ŀ¿������������������Ŀѿؿܿݿ�ݿѿ�������� ����(�7�A�D�G�C�A�5�-���6�+�1�5�6�B�F�I�F�B�6�6�6�6�6�6�6�6�6�6�	� ������� �	��"�/�2�:�;�>�;�/�"��	�	����������	��"�.�3�4�0�%�"���	����z�s�g�s������������������������������������ûǻû��������������������Z�T�M�N�W�g�������������������������g�Z�������������|������������������������������������������
��#�+�<�@�=�0�#��
��ŹŭŔŇ�{�j�e�{ŔŠ������������������Źùöù������������������ùùùùùùùù�������������������������������������������������)�B�O�[�i�d�g�[�O�B�6�)��+�5�0��������0�=�I�T�`�z�{�u�b�I�=�+�x�x�l�d�_�X�S�P�F�@�F�S�_�l�r�x�������x�������������������Ŀɿ̿ѿӿտѿʿĿ����ù¹������ùϹܹ������������ܹϹú@�'����'�;�B�L�Y�e�r�w�y�w�r�j�Y�L�@�����ݽнŽнԽݽ�������&� ����������������ûлٻһлû����������������_�W�S�F�B�J�S�_�l�����������������x�l�_����������������������������������������Y�U�Y�_�f�g�q�r�~�|�w�v�r�f�Y�Y�Y�Y�Y�Y�:�5�:�?�G�S�T�T�S�G�:�:�:�:�:�:�:�:�:�:�`�W�Z�]�`�l�y�����y�p�l�`�`�`�`�`�`�`�`E*E*E*E6E7ECEPE\EiEfE\EPECE7E*E*E*E*E*E*E�E�E�E�EuEjEuE|E�E�E�E�E�E�E�E�E�E�E�E� V / Q % J X J 8 G ( L L $ - � S T t ; 7 o 8 2 O < 4 Z j B ; ? ^ l B u _ H w N 3 G 4 2 Q  a 1 = g \ 5 T n ? ` f v B l P U N E A T ~ v ? R c H    s  &  ~  �  @  �  ^    p  |  �  �  �  �  �  �  �  �    �  D  �  "  �    �      F  d  #  �  �  
  �  <    :    �  �  �  �  A  �  �  8  �  R  ^  �  +  �  j  q  �  �  I  v    F  �  �  i  �  c  �  $  �  �  �<���<49X<�C���1;��
��`B�t��+�D���ě��ě���O߼�o��t��u�49X�'����P���
��^5���㼋C��o�ě���P��`B��1���ͽ�{��`B��h�]/�o�t��t������<j�u�<j�t��t��49X�D�����8Q콇+�,1�49X���
�Y��ixս��P�T����C����P��E���O߽�O߽�C����T�����o��C����w��j�����j��xվI�B�B˅B+NB7vB1�B�PB
c�B�gB�ZBĊBk�B&8�B
HB�^Bh�BU:B ��A�eJB�!B$�+B,O�B]�B_A��_B!��B�B�;B��B9B��B}�B�B+9�B,_B�:B�B�&B7?B#��B
��B�_B�*B2BZ=B��B�A�e�B	� B!��B�QB�NB�BC�B
]_B�B��BE�B^B!t�B�;B�nB�B>&B%DNB&r�BNBRYB��B��B�=B��B��BÓB0�Bz\B��B@QB
@�B�\B��B�>BO0B%�kBRB��B=xB@^B şA���B�SB$�B-?TBGvB7A��B!�9B��B��B�$B,tB��B%BB��B*�B<�B��BA�B#uB�3B#xB
�-B�"B�nB@;B�BѺB�)A��=B	�"B!�<B�B��B��B>iB
B��B�QB9�B�B!AB��B�+B?�B?�B%2�B&u$B�zB�IBýB�cB=�B�A�Q@��#AƋ8A�)�A�t�B	�SA��A��:?��:A��A�o�A�8�A>	<A̕kA�&mA���@���B7K@�Z�@���A,A� >A�.A�}?؄�B�EA�ϼA� �AĭC@\��A�'yA!��Ah��A��?A��lA�n�A��(A�h�A"8�A�=�A���A���A�7�Aw�WA�mA׾A���A[AE�v@�DA���A�S/A��A��A�AиA�jxB
ˍ@�pAw��>� �?���A/sz@��1@�NyA婺@��mA�$AC��_C�0�A㏃@��A�&uA�nTA�{�B	�NA�A�i?���A���A�T�A�Y�A>�YA̋A�v�A��@ݑXB��@���@�9A��A���A��A���?���B;kA���A׃-AĿ�@\`A�.!A �qAi%�A���A���A���A�u�A��A#8A��A�~A���A���Av�A�~ Aג�A�5A[�AE@�S�A�{�A���A�{`A��AνxAϥA֕sB	Ϋ@��Aw�>���?��HA0�.@��*@���A��@� _A�vA�*C���C�*$            "            &            G               $   
      
   Q                     B      @                              !      	                           &                     $                     	      	   
         -                        /            E               #            O               !   #   +      /         '                     !                                 !         '            -            !                                                               =                           ;               !   #         /                              !                                          '            %            !                           N?A<O_�Na~�O��tNz�O�O�OF��NNXO22�N���P�JON$KN��}N��NI��NEW.O6=�N��1N��Pm��N���O�0O!�`N���O��}O��}ON9 N&�P>OaRN��OC�BN�C�N�z�O81pN�C�N�I'OP>�O�@�N�k@N���N��aN��O$�N4]�Ot�O�yN<o$N?c�O�oN��O*;PY�N6z+N���O9!�O���N�g�N�u�N�2�O�2�O&�\NkИN�3^N6��Nt`cN=PN�L�N���OE�  ]  �  �  B  �  _  �    ^  f  A    �  �  �    %  J  �  �  �  ^  �  Z    �  �  	s  �  �  n  �    |    �  �  t  �  v  B  E  �  �  �  �  �  y  N  E  �  �  �  V  �  �  g  �  �  '  7  �  �  8  0  �  �  �  �  	  �=o=C�<�j<D��<o;��
;D���e`B;�o;o:�o�49X��o�D����o���
���#�
����49X���ͼD���D����C���C����㼓t��0 ż��������j�ě��t����ͼ�����/����������`B��`B����`B��h���C��o�\)�<j��w�#�
�@��0 Ž8Q�49X�<j�H�9�P�`�aG��aG��e`B�ixսe`B�m�h�u��%������1��{��{��vɽ\7<HKJHE<<;2277777777nz|����������zwonnnn����
��������������������������������GHUanz{zvnaUMHGGGGGG��������qt�������������xtomq������
���������Z[hkt�trh[ZUZZZZZZZZ	
#*/9<CD></#	���������������������#0b{�����bU<0������
#'/50/#
���������������������������������������������KO[hqqh[QOKKKKKKKKKK��������������������HMT^jmqxrmjfca^THEEH��������������������

 #).#








w������..&�����}tw����������)5=BEHHB5)CHKTahmnnmkgdaTPHDAC������������������������������������������������������������ACFIJNY[agjttpg[NBAA #/<@<:/#"          ����������������������������������������ttu�����������tjkmtt��������������������)+.,,)) ).569:85)'��������������������BHU\anpznkaUSHG@BBBB������������������������
##
��������dgt�������������m]^d������	�����������?HJUafeaa[UJH<??????������������������������������������������������������������#/<>=</#imtz�������zumjfefiiT[]gtv������tng[[UTT��������������������IO[]]\[YOLIIIIIIIIII��)25+)�������DOT[hotz�~yth[OMDDDD��������������������Z^n}����������tgYRWZ��������������������������������������������������������������������������������������������������SUUaknz����zsna[UQSSwzz�������������zvwwz����������������~zz��	#&'&# ��#+//-#.00;<IJMOOMI<110....�������������������������������������������������������������������������


#$/29<7/#
ĿĸķĿ����������������ĿĿĿĿĿĿĿĿ���������������ʼּ����ټּʼ��������a�Y�U�M�N�U�]�a�n�x�y�n�a�a�a�a�a�a�a�a��������������������%�3�:�?�/��
����������������������������������������������������������$�0�=�I�N�I�H�=�4�0�$��ѿпϿѿؿݿ�����
���������ݿ������������������������������������������e�b�b�e�n�r�|�~�����~�r�e�e�e�e�e�e�e�e���������������������������������������H�E�;�2�;�=�H�T�a�m�q�o�m�a�U�T�H�H�H�H�����W�M�G�E�N�`�����������������������4�4�/�3�=�A�M�Z�f�g�p�s�u�w�s�q�f�Z�A�4àÝÛÝàìù����ûùìàààààààà�U�L�H�F�H�N�S�U�a�e�a�Z�X�W�U�U�U�U�U�U�T�Q�K�M�T�a�k�j�a�]�T�T�T�T�T�T�T�T�T�T�f�c�Y�M�L�K�M�Y�f�l�r�l�f�f�f�f�f�f�f�f����ƭƧƚƘƚƳ��������������� �������ʻû»��������ûлۻܻ�ݻܻлûûûûûü��������������żǼż����������������������������������߽��!�:�A�A�!���ּʼ��������z�|�������������������������������N�J�C�F�N�T�Z�g�s�~�~�z�}�s�i�g�a�Z�N�N�(� �����%�(�5�8�A�N�Z�c�e�Z�M�A�5�(�L�J�F�@�@�9�@�L�Y�e�h�r�z�~�r�e�[�Y�L�L������������Ƹ�����������"�$���������������������������*�6�C�C�6�/����6�)�����������B�D�O�V�\�[�O�O�6�H�B�@�@�H�Q�U�Z�V�U�H�H�H�H�H�H�H�H�H�H�ɺ����������ֻ�!�:�S�b�_�U�F�-�!�������	����������#�)�+�1�4�)����������������������������������������������T�N�G�D�>�<�>�G�T�`�b�t���������y�m�`�T�����������������������������������������t�q�p�t�{āĉčĚĦĳĵĴĳĩĦĚčā�t�H�?�?�@�C�H�J�P�T�a�c�m�o�s�t�s�h�a�T�H�������������������������������������������������������������������������������������������������������ĽƽĽ����������m�i�^�W�W�T�V�a�z���������������������m����������������������������������������������������������������������������������������*�/�6�6�?�6�*�������Ŀ¿������������������Ŀѿؿܿݿ�ݿѿ������������(�3�5�A�D�A�A�5�(���6�+�1�5�6�B�F�I�F�B�6�6�6�6�6�6�6�6�6�6�	� ������� �	��"�/�2�:�;�>�;�/�"��	�	�������������	��"�'�&�"����	������z�s�g�s������������������������������������ûǻû��������������������Z�Q�Q�Z�k�x�������������������������g�Z�������������|���������������������������������������������#�(�<�>�;�0�#��
��ŹŭŔŇ�{�j�e�{ŔŠ������������������Źùöù������������������ùùùùùùùù�������������������������������������������������)�B�O�[�i�d�g�[�O�B�6�)��=�4�7�0���������0�=�I�S�_�x�z�t�b�I�=�l�e�_�X�S�R�M�S�_�l�q�x�{���������x�l�l�������������������Ŀɿ̿ѿӿտѿʿĿ����Ϲùù��������ùϹܹ����� �����ܹϹϺ@�'����'�;�B�L�Y�e�r�w�y�w�r�j�Y�L�@�����ݽнŽнԽݽ�������&� ����������������ûлٻһлû����������������l�b�_�S�O�Q�S�_�l�x���������x�s�l�l�l�l����������������������������������������Y�U�Y�_�f�g�q�r�~�|�w�v�r�f�Y�Y�Y�Y�Y�Y�:�5�:�?�G�S�T�T�S�G�:�:�:�:�:�:�:�:�:�:�`�W�Z�]�`�l�y�����y�p�l�`�`�`�`�`�`�`�`E*E*E*E6E7ECEPE\EiEfE\EPECE7E*E*E*E*E*E*E�E�EE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� U / Q % m X E ' G ( L I $ + � S > t , 7 Z 8 2 A < 4 Z I B 6 ? ^ J B d a H w B 3 ; 4 2 Q  a 1 4 g \ ( T i ? ` f v D f P G N E A D ~ v ? R c 8    Y  &  ~  �  �  K  ;  �  p  |  �  .  �  �  �  �  g  �  �  �  l  �  "  f    �    �  F  9  #  �  �  
  )  �    :  �  �    �  �  A  [  �  8  E  R  ^  >  +  y  j  q  �  �  5        �  �  i  �  c  �  $  �  �  �  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  Q  Y  \  [  T  K  ?  .    �  �  �  [    �  F  �  i  �  t  �  �  �  �  �  �  f  A    �  �  �  G    �  `    �  �  _  �  �  �  �  �  �    {  v  q  l  g  c  ^  Z  U  R  S  T  U  B  2  #       �  �  �  �  n  ?  
  �  �  O    �  �  "  m  �  �  �  �  �  �  �  �  �  �  �  �  {  j  X  F  $   �   �   �  E  L  S  Y  ]  ]  T  E  1    �  �  �  �  ]  1  �  �  |  O  �  �  �  �  u  ^  E  &    �  �  �  S  )    �  �  w  +   �  �  �  F  �  �  �  �  �        �  �  �  �  q  5  �  �  I  ^  U  L  C  8  -      �  �  �  �  �  �  f  K  0    �  �  f  `  U  H  ?  5  '      �  �  �  �  S    �  @  �  �   �  A  :  3  +  !        �  �  �  �  �  �  �  �  �  {  o  d  �  �        �  �    -  �  �  �  �  O  �  |  �  h    �  �  �  �  �  �  �  �  e  F  %     �  �  �  ^  >    �  �  �  �  �  �  �  �  y  q  b  H  '    �  �  k  -  �  �  J  �  �  �  �  �  �  �  �  �  �  	�  	�  	�  
  
2  
Q  
p  
�  
�  
�  
�          
    �  �  �  �  �  �  �  �  �  l  X  D  .      Z  �  �  �  �  �  �  �  �  �  �  �  
  %    �  �  �  W  X  J  D  >  7  0  +  &    �  �  �  �  `  4    �  �  y  :   �  �  �  �  �  �  �  �  �  �  �  |  O    �  �  )  �  >  �  B  �  �  �  �  �  �  �  �  �  x  d  I  $  �  �  �  U     �   �      �  �  �  �  k  <    �  �  �  �  �  S  �  {  �  �  3  ^  Y  U  N  ?  0      �  �  �  �  �  �  �  �  �  �  i  L  �  �  �  �  �  �  �  �  �  �  �  �    q  d  V  >  $  
   �  A  U  Z  X  S  N  H  @  5  $    �  �  �  �  b  4  �  �  :        �  �  �  �  �  �  �  �  �  �  �  �    g  F  "   �  �  �  �  �  �  �  v  W  5    �  �  �  �  \  %  �  �     �  �  �  �  �  �  �  �  �  �  v  f  Q  @  1      �  �  �  u  ^  |  �  �  �  �  �  	s  	b  	R  	1  �  �  C  �    �  �  �    �  �                           �  �  �  �  �  �  e  �  ~  f  D    �  �  r  .  �  �  �  h  0  �  a  �  f   9  n  g  a  Z  T  M  G  @  9  1  )           	        !  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  X     �   �  �  �  �  �  �  �  �    	  �  �  �  �  �  b    �  V  �  9  |  w  r  m  g  a  Z  R  K  G  E  B  <  6  +      �  �  �  �       �  �  �  �  �  �  �  �  j  M  9  (        �  �  �  �  �  �  �  �  �  �  y  h  R  :        �  �  e  G     �  �  �  �  �  �  t  g  Y  L  >  0  "      �  �  �  �  �  �  t  s  q  p  l  ^  O  A  4  )        �  �  �  �  �  R  "  �  �  �  �  �  �  �  �  v  _  D  &     �  �  p  7  �  �  �  v  X  N  J  I  D  6  $    �  �  �  z  I    �  �    S  i  �  2  ?  B  =  1  $      �  �  �  �  �  �  ;  �  �  %  �  E  ?  :  2  *         �  �  �  �  �  b  ;      �  �  �  �  �  |  v  m  c  Z  R  I  :  *      �  �  �  �  k     �  �  �  �  �  �  �  �  t  V  6    �  �  �  �  �  k  A    �  �  �  �  �  �  �  �  �  �  y  a  D    �  �  z  )  �  n  
  �  �  �  �  �  }  o  a  R  C  3  !    �  �  �  w  D    �  �  �  �  �  z  k  Z  G  4    	  �  �  �  �  o  ?    �  w  :  Q  c  p  v  x  q  c  J  +    �  �  h     �  a  �  c  �  N  C  7  ,       
  �  �  �  �  �  �  �  �  l  W  B  -    E  ;  2  )                   !  (  0  1  /  .  -  +  �  �  �  �  |  o  Y  <    �  �  �  {  P    �  r  �  �   �  �  t  g  W  F  1      �  �  �  �  o  M  '     �  �  Q    �  �  �  �  �  �  �  h  O  9  %    �  �  ~  0  �  �  ~  >  V  Q  >  "    �  �  �  s  J  4    �  �  c    �  h  �   �  �  �  �  �  �  r  ]  C  '    �  �  �  �  }  `  G  ?  8  0  �  �  �  k  B    �  �  �  J    �  �  T  0  �  �  �  ~  E  g  \  V  >    �  �  �  �  �  ^  /  �  �  �  V    �  �  K  �  �  �  �  �  q  <    �  �  �  �  \  �  3  �  '  �  s  �  �  �  �  �  �  �  �  �  r  U  <    �  �  O    �    �  �  '    �  �  �  �  �  �  �  n  ^  L  ;  -      �  �  �  �  7  7  5  /  (         �  �  �  �  b  .  �  �  t  1  �  �  �  {  o  b  T  >    �  �  �  S    �  �  E  $  �  8  b   }  �  �  �  �  �  �  �    h  R  <  )    �  �  �  �  ]  1  �  8  0  '          �  �  �  �  �  �  �  �  �  �  q  ^  K  &  %  %  &  '  )  ,  /  (        �  �  �  �  �  �  �  v  �  �    x  r  k  e  X  J  ;  ,          �   �   �   �   �   �  �  �  �  y  T  0    �  �  �  �  �  �  N    �  �  �  �  {  �  �  �  �  �  h  M  1    �  �  �  �  �  �  e  K  1    �  �  q  b  R  B  1      �  �  �  �  �  �  t  k  b  Z  S  K  	  �  �  �  �  t  Q  )  �  �  �  K    �  �  �  R  �  �    �  �  y  `  G  ,    �  �  �  �  �  i    �    }  �  $  �