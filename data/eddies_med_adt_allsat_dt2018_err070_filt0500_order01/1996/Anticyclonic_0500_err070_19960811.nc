CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��hr�!      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N½   max       P�7      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��C�   max       =�l�      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E��
=q     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(�    max       @v|z�G�     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @P�           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�S�          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ix�   max       >r�!      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�,�   max       B-\�      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B-Qd      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?!��   max       C��g      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?('   max       C���      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�O�   max       P7�O      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ح��U�   max       ?�p��
=q      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       =�      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E�\(�     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?У�
=p    max       @v|z�G�     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P            t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @�O           �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�@N���U   max       ?�o���     �  QX                     $         3      	                  	      �   b   .                                        1      @            �      �   )      "         1         0            '    NR�eNI�&O�(&N�H�N/WRN77cO�ZVN�'�Nf��P|��N��O$N½O&��N�$Nɖ�OB�O��Nm��P�!�P�7P NUe�O��fNʵ�N�kP5��O��NK_KP�O՝�NǞO}�NA�,O�cGN���P]egO�A�O��O�G0PP�?N���P8�O�,N(pOC &N��N��P��O#BnO��O��1OU�eNkD�O%��OT�O`�ؽ�C��D���u�#�
�#�
�o��o�D���D����o%   %   %   ;D��;ě�;ě�<o<#�
<#�
<49X<D��<T��<T��<e`B<�o<���<��
<�1<�9X<���<���<�h<�h<�h<�h=o=+=t�=��=�w=8Q�=<j=D��=H�9=H�9=P�`=T��=aG�=e`B=ix�=�o=��=�hs=���=�-=�^5=�l�-*/<DHIHF<7/--------��������������������!.<IUZ`hnkbUI0#XTSTR[cgpsqlg[XXXXXX��������������������������������"&/<HNUacwqiaU</+%%")58=75.)!|}���������||||||||�����
0IP[[O<#
�����vz���������������zvv�)696/-)(�4.-6BFEB864444444444"#,09IUY\XUIC0)#"���������������������
#01330(#
�	"/;@B<;3/("	��������������������rooty������trrrrrrrr)9g��������}tgN:��������;VXUB)�������������������������������������������>9;B[ht�������th[OB>{vwv����������{{{{{{�������������������������� ������������������� �����������"��������/5NgmqrngN5(�ZXZblz�����������zaZ��������������������)6OV[e[OB6)#;;BO[[b[SOKB;;;;;;;;xuxx}��������������xgortu�����������{thg��4598=MMHB5�����������������������
#/<FF>83/#
�����)/<;2,����������)1*$������\bhnv{�������{nb\\\\������
"#"
�����+('(/7<HUnz��zfUH<1+hjknz�{znhhhhhhhhhh����������������������������������������)*)(%!32:Tm���������zm];3 $).66BHOTTOKB>6)������� $ ����������&+--+#�����TQPPT[htw}|}����th[T?<BNR[]^[NEB????????�����
#%'(##
��������������������
���������
 !" E\EiErEtEiE^E\EXEPEEEPEYE\E\E\E\E\E\E\E\�����
���
���������������������������񻅻������������������m�_�Y�_�]�V�_�l�x������������������������������������������������������������ùù÷ù����������������)�2�1�5�)�#����������������� �������������������������������ʾʾѾ;ʾ��������������������������������������������y�x���������A�M�s�z�e�k�`�]�A�(����ֽܽ�����(�AFFF$F'F1F6F:F1F)F$FFE�E�E�F	FFFF���������������������������z�y�s�r�y�z���Ŀѿݿ��ݿѿ˿Ŀ��ĿĿĿĿĿĿĿĿĿļr���������������������r�n�h�f�b�f�p�r���������������������������������������Ҽּ����������������߼ּ̼мѼּּּ��H�T�a�f�m�t�x�x�m�a�T�R�H�C�;�3�4�;�>�H�B�N�P�U�T�T�N�B�6�2�������)�5�>�B��������	��	��������������������������āčğā�n�F�6�)��������õò�����Bā��������/�;�V�b�`�M�/�	�������������������������������������������g�Z�N�s�x�����ûŻллٻѻлû����������ûûûûûûý�����������нĽ����������Ͻؽ���zÇÓàæàß×ÓÇ�z�w�t�v�z�z�z�z�z�z����'�,�3�4�7�4�'� ���������������(�E�H�G�A�(������ѿ����������Ŀ�����������������ϹĹ��ŹϹ�'�'�3�=�7�@�G�@�3�0�'� �!�'�'�'�'�'�'�'������!�#����ѿĿ����������ѿؿݿ����(�5�A�H�J�Q�\�a�a�Z�N�A�5����� ��<�H�U�a�h�n�a�U�H�<�/�&�/�0�<�<�<�<�<�<�ʾ׾������	���׾ʾ������������¾ʺ�'�+�,�'�%���������������ʾ׾�	�"�.�=�D�C�;�"�����׾þ������ʺ3�@�L�V�Y�e�e�n�r�{�r�e�Y�L�@�<�3�-�1�3���������� �����Ɓ�u�\�N�>�2�3�;�O�h���Z�s���������������������N�5�+�)�2�A�N�Z�����"�*�.�;�E�A�4�"������޾Ҿ;ܾ���#�0�<�I�M�M�K�I�<�0�#��
���������
��ûܻ������������ܻû������}�}�������ú�� ������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DoDkDrD{D�D��4�A�M�Z�k�s�t�s�l�\�G�A�4�2�(� ���(�4�����������������������������������������ʼ޼���������������ּӼʼü������ʾ4�7�A�M�P�M�F�A�4�(�������%�(�1�4�����������������������������������������tčĦľ����ķĭİīĒĒčă�h�L�I�[�i�t���������úɺֺֺ�ֺĺ������������������`�l�y�����������{�y�l�`�S�J�F�C�G�H�S�`��������������������¿²¦¡®²º¾�˼���'�4�@�M�H�@�4�'�������������b�o�z�s�o�e�b�V�O�M�V�[�b�b�b�b�b�b�b�b�{ǈǔǡǩǭǨǡǚǔǈ�{�o�n�k�n�o�p�{�{EuE�E�E�E�E�E�E�E�E�E�E�E�EuEtEiEuEuEuEu�I�G�I�R�U�b�n�{ŇňŐőŎňŇ�{�n�b�U�I P 6 + H m Z 4 < 3 G K O 8 G 2 F = G " \ < D x M 6 I _ A k * b Z : F + � e a ; 4 6 h 4 K 9 ; K / X A ^ ) > Z  E %    P  Z  �  �  �  s    �  �  >  �  >  ,  u  ;  �  �  s  v  w  �  �  �  �  �  �  �  �  �  �  P  �    i  ;  3  �  �  j  �  �    �  �  >  �  
  �    [  �    �  h  _  .  ɽixս'�<t��D��;�o;�o=o;o;o=L��;��
<o;o<e`B<T��<�C�<�/<�t�<�o>r�!=�G�=m�h<���<�=t�<���=L��=<j<���=e`B=ix�=\)=T��=t�=��-=,1=ě�=�%=��=�O�>>v�=�o>fff=�E�=Y�=�1=�%=}�=�
==���=�^5=�l�=ȴ9=� �=�/>�>z�B��B�B&RqB	"�BfB��B��B��B)��B$��B�B��B��B&*�B�B%aA�,�B0HB��B�\BB*aB!�B�qB
�,B"z2B`B��B�qB�BTGB$.B��B��B 	�B��B^�B�B�3B��Bk�B(��B�BwB��B�0B��B�(A�&nB�5B-\�Bg�B��BB;B�B�HBk.B�~BĪB&B�B	=�B��BE(B�bB��B)�xB%�*BW�B�B�PB&8eB�(B%>�A���BIWB�HB�B��BKB"CB�xB
�}B"?xB{hBZ�B�|B�<BC�BB�B�aB��B ?vB¥B��B�	B��BA�BPWB(�BB�B=}B3�B5�B�B�A�b.B4`B-QdB��B�xB?�BƏB��BA_C�˧A�Q@��EA�]A��A�.�AҲAN�@�rUA5k�C��gAp��A{��@�)A�$pA�A��ZA�qWA��UA�C A�j4A��7@���A-�A�߇@���A�wd?!��?�u%A~�2A���AĺFAT,j?�;5AY��?�x�B�nA���A[)"A�R�@�[@X��C��A<vA�{HA�A8�AA�%A��@&=�AA���@Ō{BG�B�@C�sA�oC��bA�~�@��A��RA��A��AҒgAN� @�@�A39C���Aq�A{$@���A�f�A
fA��KA�jLA��AԀIA�2�A��L@��8A-)Aɗ9@�5A��H?('?�r9AI|A�kA�\ASc?�&dAZ�1?�� B��A�̔AZ��A�4k@��@[��C��A<��A���A�A9l*A�?A���@+LfA�A��F@��B:�B7C�2A��0                     %         4      	                  	      �   c   .                            !            1      @            �      �   )      "         2         1            (   !         !                     7                              G   =   +               5   #      '   #            #      5   %         -      '                  -                                                      )                                 /                  '         !   #                  )   %               !                                          NR�eNI�&OlNt�N/WRN77cN�"�N�'�M�O�P	�N��N��gN½O&��N�$N>�(N�&O��Nm��O�B�P7�OO���NUe�O]��N�N�N�kO覱O26NK_KO��sO���NǞO=jNNA�,O� rN��RO�`O�A�O��O�
Oڴ�N���O�?�O�	�N(pOC &N��)N�A�O�Y�O#BnO��O��OU�eNkD�O%��OT�O`��  K  �  @  �  x  �  =  �  �  �  �  �  7  \  F  �  �  e    Z  Y  x  A  (  i  �  �  B  �  q  �  �  �  �  �  �  <  �  �  �  �  a  J    �    �  7  h  -    L    .  �  
�  ؽ�C��D���#�
��`B�#�
�o<T���D����o<T��%   ;o%   ;D��;ě�<t�<e`B<#�
<#�
=�=��<���<T��<�t�<���<���<�`B<�<�9X=o<�h<�h=+<�h=0 �=+=P�`=t�=��='�=\=<j=�-=aG�=H�9=P�`=]/=e`B=�\)=ix�=��=�O�=�hs=���=�-=�^5=�l�-*/<DHIHF<7/--------�������������������� &0<ISU[``\UI<0# XX[dghmmhge[XXXXXXXX��������������������������������../6<CHRUYUQH<0/....)58=75.)!�����������������������
#4<ELLH<0#
�����vz���������������zvv	))*/++)!4.-6BFEB864444444444"#,09IUY\XUIC0)#"��������������������#-/$#		"/4;3/("
	��������������������rooty������trrrrrrrrB?=?CN[gt{~~ztpg[NGB����)6GIB6)�����������������������������������������������=@BO[ht����th`[OB=}yyz����������}}}}}}���������������������������
��������������������������������"�������"5BN[agfjjg[B5)#^]ahnz�����������zb^��������������������)6DMOUUOGB6));;BO[[b[SOKB;;;;;;;;��������������������ipstv���������ytiiii)128BF@5)	��������������������
#/<FF>83/#
������)685/*�����������������\bhnv{�������{nb\\\\�����

�������.+*,2<HUakqxq\UH<6/.hjknz�{znhhhhhhhhhh����������������������������������������
&$ ;>CPamz�����}zmaTG>; $).66BHOTTOKB>6)��������#����������$*,,) ����TQPPT[htw}|}����th[T?<BNR[]^[NEB????????�����
#%'(##
��������������������
���������
 !" E\EiErEtEiE^E\EXEPEEEPEYE\E\E\E\E\E\E\E\�����
���
����������������������������l�x���������������������x�l�i�f�e�_�g�l����������������������������������������������������������ùù÷ù����������������)�2�1�5�)�#���������������������������������������������������ʾʾѾ;ʾ����������������������������������������~������������4�M�\�^�W�M�E�4������ݽ�������4FFF$F'F1F6F:F1F)F$FFE�E�E�F	FFFF���������������������������y�w�y�z�������Ŀѿݿ��ݿѿ˿Ŀ��ĿĿĿĿĿĿĿĿĿļr���������������������r�n�h�f�b�f�p�r���������������������������������������Ҽּ�������ּԼԼּּּּּּּּּ��T�a�a�i�m�o�p�m�a�T�P�H�A�?�H�I�T�T�T�T�B�N�P�U�T�T�N�B�6�2�������)�5�>�B��������	��	�����������������������������)�6�B�N�O�L�A�6�)����������������;�H�T�U�J�0�&�	��������������������/�;�����������������������������������������ûŻллٻѻлû����������ûûûûûûý�������
����ݽнƽĽ��Žнսݽ���zÇÓàäàÝÓÓÇ�z�z�v�x�z�z�z�z�z�z����'�,�3�4�7�4�'� ����������������(�=�C�A�5�(�������ѿ����ѿۿ��ܹ��������������ܹϹιʹϹйܺ'�'�3�=�7�@�G�@�3�0�'� �!�'�'�'�'�'�'�'��������������ݿѿȿ������Ŀۿ����(�5�;�B�F�L�V�\�]�T�N�A�5�������<�H�U�a�h�n�a�U�H�<�/�&�/�0�<�<�<�<�<�<�ʾ׾���������׾ʾƾ����������ƾʺ�'�+�,�'�%�����������������	��"�.�5�6�-�"��	���׾Ѿ̾˾վ�3�@�L�S�Y�d�e�l�e�Y�R�L�@�?�3�/�3�3�3�3�h�uƚƧ������������ƧƎ�u�\�J�C�C�G�O�h�Z�s���������������������N�5�+�)�2�A�N�Z�����"�*�.�;�E�A�4�"������޾Ҿ;ܾ���#�2�<�F�I�L�K�I�<�0�*�#��
�������
����ûлܻ�����ܻлû������������������ ������������������������D�D�D�D�D�D�D�D�D�D�D�D�DxDuD{D�D�D�D�DӾ4�A�M�Z�f�p�p�h�a�Z�M�A�4�(�$� �#�(�2�4�����������������������������������������ʼ޼���������������ּӼʼü������ʾ4�A�J�M�N�M�C�A�4�(�����(�+�4�4�4�4����������������������������������������čĚĦĳļĿĽĳĦĚđā�t�n�f�l�o�tāč���������úɺֺֺ�ֺĺ������������������`�l�y�������������y�l�`�S�J�F�D�G�I�S�`��������������������¿²¦¢£ª²¶¿�˼���'�4�@�M�H�@�4�'�������������b�o�z�s�o�e�b�V�O�M�V�[�b�b�b�b�b�b�b�b�{ǈǔǡǩǭǨǡǚǔǈ�{�o�n�k�n�o�p�{�{EuE�E�E�E�E�E�E�E�E�E�E�E�EuEtEiEuEuEuEu�I�G�I�R�U�b�n�{ŇňŐőŎňŇ�{�n�b�U�I P 6 % F m Z # < = 2 K D 8 G 2 = A G " 0 : * x @ 3 I T % k  f Z . F ) � _ a ; 0 1 h 4 ) 9 ; 9 3 * A ] ! > Z  E %    P  Z  �  =  �  s  �  �    �  �  �  ,  u  ;  _    s  v  n  8    �  �  �  �  ]  �  �  �  
  �  �  i  1  �  �  �  j  ]      �  �  >  �  �  �  �  [  z  1  �  h  _  .  �  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  K  M  O  Q  S  S  O  I  8  &    �  �  �  �  p  ;  �  �  b  �  �  �  �  �  w  h  T  @  "    �  �  �  b  5     �   �   w    #  1  <  @  :  1  '      �  �  �  �  �  p  [    �  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      %  8  K  x  S  A  1    �  �  �  �  a  <    �  �  �  �  V  *  �  �  �  �  �  �  �  	  /  R  r  �  �  �  �    6  S  p  �    �  �    L  {  �  �  �    6  =  5    �  �  `    �  A  �  =  �  �  �  �  �  �  �  �  �  �  �  }  l  X  D  0     �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    !  0  s  �  �  �  �  �  �  �  �  �  m  8  �  �  {    n  �  c  �  �  �  �  |  a  F  +    �  �  �  �  a  ?     �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  i  Q  6     �   �  7  6  5  4  3  3  2  1  0  /  *  !         �   �   �   �   �  \  Y  V  O  H  A  :  2  )  !    	  �  �  �  �  �  {  ~  �  F  A  <  7  0  *  !         �  �  �  �  �  m  >    �  �  e  m  w  �  �  �  �  �  �  �  l  ^  V  O  H  @  7  .  %    D  \  o  �  �  �  �  �  �  n  O  -    �  �  �  {  Z  3  �  e  T  C  ,    �  �  �  �  �  �  �  �  �  �  y  d  J  &      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
�  �  /  �  �  �  v  �  L  Y  :  �  d  �  �  �  -  �  �  �  @  �  �  .  M  Y  R  D  +    �  �  �    �  �  J  �  �  |  �  �  '  R  n  x  s  b  C    �  �  h    �  [    �  
  �  A  E  J  N  I  B  ;  2  '        �  �  �  �  �  �  �  �  �      %  (  &      �  �  �  �  �  ^  4    �  �  3    \  e  h  h  g  a  X  K  7      �  �  �  V  "  �  �  )  �  �  �  �  �  �  �  y  o  d  a  _  ]  [  Y  [  a  g  o  w  �  i  �  �  �  �  �  �  �  |  R  #  �  �  �  �  n  )  �    g  �         ,  ;  @  B  >  5  #    �  �  {  >  �  n    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     ;  T  g  p  p  k  `  O  8    �  �  �  5  �  �    q    i  v      s  [  ;    �  �  q  �  y  ?  �  �  ;  �  �   ^  �  �  �  �  �  �  �  �  �  �  v  e  U  E  6  '    /  @  P  =  b  z  �  �  ~  n  V  ;    �  �  �  v  ?  �  �  o    �  �  �  �  �  }  v  n  f  ^  T  J  ?  6  -    
  �  �  �  R  /  f  �  �  �  �  �  �  �  �  y  M    �      y  �     �  �  �  �  �  �  �  �  �        �  �  �  �  �  v  Z  7    _  �  �    .  ;  7  $    �  �  l  �  �  �  j  �  E  �  �  �  �  �  �  �  �  m  L  &  �  �  �  j    �  O  �  �  E    �  �  �  �  �  p  P  +    �  �  p  =    �  �  k    �  v  �  �  �  �  �  �  �  �  h  @    �  �  v  A  
  �  �  9  �  �  0  �    J  t  �  �  }  Y    �  ,  �  
�  	�  �  "  �  �  a  Q  @  1  #      �  �  �  r  3  �  �  �  �  <  �  �  E  �  �  �  -  I  E    �  u  �  \  �  �  �    &  �  �  	=    �  �  
        �  �  �  X    �  g    �  _  �  �    h  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  l  ]          �  �  �  �  �  e  ;    �  f    �  3  �  �  8  �  �  �  �  �  �  �  �  �  �  �  H     �  }  <  �  �  Y   �    "  /  6  1  ,  &         	     �  �  �  �  �  �  �  �  Y  �      Z  f  d  S  6    �  �  Z    �  /  �  (  Y  �  -  "      �  �  �  �  y  P  #  �  �  �  f  6  �  �  R  n      
  �  �  �  �  q  4  �  �  f    �  v    �  �  j  �  �  7  J  A  4       �  �  �  g  )  �  ~    j  �     -  E    �  �  �  �  L    �  �  �    �  �  �  |  ]  .  �  �  I  .      �  �  �  �  �  �  u  L    �  �  C  �  �  +  �  a  �  �  �  {  _  A  #    �  �  �  f  (  �  �  g  "  �  �  &  
�  
�  
�  
s  
W  
6  
  	�  	�  	�  	J  �  g  �  �  �    	      �  �  ^  )  �  �  T  	  �  h    �  V  �  \  �  =  �  ;  �