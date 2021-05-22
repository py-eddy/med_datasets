CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�9XbM�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�
R   max       P�u�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �T��   max       >�      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Y�����   max       @EB�\(��     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vN=p��
     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @N�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ȝ        max       @�V@          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �\)   max       >A�7      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�B�   max       B(��      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�n    max       B(=      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?<w�   max       C�Uq      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?M%�   max       C�W�      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�
R   max       PC��      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��4֡a�   max       ?�z�G�{      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �T��   max       >�      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?fffffg   max       @EB�\(��     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @vN=p��
     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @N�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ȝ        max       @�          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�R�<64   max       ?�z�G�{     �  M�      M      ~            !   1                        &   E         
         �                  j   5      +         
   
   /   -         
                        H   2      8N��P�u�N���Ps+LO��O$%TN^�}O��P&N�q�O�H�O`1�NAT�O��N�W�M�?OËcP��Ng��N���N�O$��N+��P'�N-��O;7�Opi�O�H�M�
RP\�P3kO46�O���OxON�͓N��lN|}O��O��N�:�N$��N�-O��N�<O)��O(NN�sN�RSOB�O�EO��)N(1Oh'�T���u�u�t�:�o;ě�;ě�<o<t�<#�
<D��<u<�t�<���<���<��
<�9X<�9X<���<�/<�<�<��<��=+=+=\)=t�=t�=�P=#�
=,1=0 �=<j=@�=D��=T��=Y�=]/=}�=}�=��=��=�7L=�7L=�hs=���=��
=��
=��
=��=��>�pqtz��������xtpppppp#<IUanu~|p_</#-///-)/<HSUYUUQH</--��������
�������������

������A@;8?BN[gjsqge\[NBAA���	 ����������������������������������������������������������������DB??HTamz������zmaOD	
#/<HU]^UI</&	|��������|||||||||| "(5BKNR^bd[NB),()012<AIKJIGA<0,,,,A>@BOPQPOBAAAAAAAAAA����
$/<GMNMHB/%�*)2<Hanz������naF<6*������
	���������������� ���������������������������������������������������������������������������BO[h��}hUROB�����������������������_`bn{�������{vsnhb_��������	
�����������)5<BBZZIB5	���������������������������
)5BN[bhig^5)�������)-ADA3)�� � 
#023410#
pkhjnt|�����������tp����������./7<HQTRH<0/........�������������������������������������������
.CILNMIB5������������������������������������������������������������������������������������������������������� 
#0<><8)#
 ����������������`amrz��zzma`\Z\````258@BNY[^eb_[TNB<522gagt����������~wrmgcadjz������������znc����������� $$�����������������zÇÍÓÙÓËÇ�z�r�n�l�n�p�z�z�z�z�z�z���	�1�:�9�;�B�B�>�/��������������������z���������������������������z�o�o�r�z�z���������'�*�%����������������������:�F�S�_�c�i�g�_�X�S�F�:�/�-�-�-�4�8�:�:����������������������������������������ù����������üùìéìñùùùùùùùù�����ùϹ�����'�0�4�0�'���Ϲʹ����������������	�/�E�P�H�/����������o�i�y��Ź����������������ŹŷŰůŶŹŹŹŹŹŹ��������������������������������������������������������������������f�s�x�y�t�s�f�c�]�a�f�f�f�f�f�f�f�f�f�f�n�{ŔŠŹ������ŹŰŠŇ�{�n�b�\�Z�[�Y�n���������������������������y�y�������������������z�~�����������m�y���������������y�`�T�G�?�;�6�;�I�`�m���5�G�S�]�a�^�N�A�5�(�������ݿҿ��ƎƚƧƳƶƳƱƧƚƎƄƍƎƎƎƎƎƎƎƎ��������������������������������T�`�e�m�t�r�m�e�`�T�G�;�9�:�9�;�G�I�T�T�"�.�4�5�3�/�.�$�"��	������������	��"�#�/�<�?�H�Q�H�<�7�/�#��#�#�#�#�#�#�#�#�ʼּ�ڼҼ������������r�M�'� �%�N�r���ʼּؼ���������������ټּּּּּֽݽ����ݽнĽ��������������������ýݾ������þƾǾƾ¾���������t�t����������s�}�v�u�w�o�f�Z�M�4�(����6�A�M�Z�f�s�ûлۻллû��������ûûûûûûûûû����)�B�xĈČĉ�~�k�[�B��������������T�`�m�����������������m�`�T�G�@�D�H�O�T�������!�.�.�!���������ּؼ�����(�5�A�N�Z�h�k�u�g�N�(���������
��	��.�;�G�T�Z�U�L�G�;�.�"��	��������	EEEEEEED�D�D�D�D�EEEEEEEEǭǡǚǔǈ�{�z�{ǇǈǔǝǡǭǵǭǭǭǭǭàâìùùùõìàÓÎÇÄÇÓØàààà�V�[�V�I�@�0����������������������$�V�<�H�U�j�r�u�~ÜèàÓ�z�a�.�#���#�3�<�N�Z�^�g�j�g�d�]�Z�P�N�A�>�<�5�8�A�J�N�N�A�N�Z�Z�Z�Y�N�A�>�;�A�A�A�A�A�A�A�A�A�A�����������������������������������������������
������
�����������������������������������������������������������޾s�w���������������z�s�f�b�Z�Y�Y�Z�f�s²¿������������������������¿·²¯¯²������������ŹŭŠŠřŠŠŭŹ�������������"�/�2�5�/�,�"��	����������	������������������������������������������仪���ûлڻ����������ܻлû��������������!�*�4�4�'�����ֺ������ºɺϺߺ������������������������������������������D{D�D�D�D�D�D�D�D�D�D�D{DvDoDnDkDoDpD{D{ , 0 - 5 * - E J x P  N [ 9 > < 4 A @ 9 & ? e S B { 2 s \ 7 ) ' L 6 % T A e _ F 8 6 ! R 5 W / ( Y V b Z 4    �  +      *  `  ~  �  ~    �  �  �  �  �    �  �  u  �  
  m  �  �  X  �  �  W   �  �  F  �  |  �  �  �  �  �    5  K  �  ]  :  {  V      �  A  �  ;  P�\)=}�o=<e`B<�h<���='�=ix�<�t�=�P='�<ě�=0 �<ě�<�9X=q��=�Q�<��=+=�w='�=t�>A�7=#�
=H�9=q��=e`B=#�
>n�=�j=]/=�{=�O�=�\)=ix�=y�#=���=ȴ9=���=�C�=���=��T=�t�=���=\=�E�=�v�=�/>��>��=�G�>>v�B
N�B�B�PB:zB#g�Bv7B��B [�B�BltA�B�B�B�Bm�B& B�iBԡBR�B�B�CB�B"kNBÇBcB'�B(��Bb�B*~B"�tB<B�%B$�hB
�uB\!B��BQ!B!ںBv{BSBF�B^�Bz�BO�B�B$�'Bj�A�kB1B
��Bw5B,�B�XBx�B
A�B��B6�B@B#�2B��B�3B��B��BH�A�n B�B2�BOB&�BSB�BkB�BD`B��B"C�B�B��B@�B(=B��B8�B"�B��B�`B%1B
��Bf�BʏBB{B">RB�)B��B>+B��B@B@�B-iB$��BB�A��^B��B
��B��BA$B�sB��A���A�~A� A��1@�3�A��&A��?<w�A��A�DA��Aқ�AA�A��8@��AG�Aj�aA��~Bp�?U	?Age�A\�=A�	�@��DA�9A&�+AJ�NA<@�CqA��Al�uA_RA���A_��C�UqB\�A�hB�-A�S�A�y�A�"�A���A�A�A�v�AD>?A��TA�t�A��fA���@�_B@Txw@.UC��yAȂ�A��RA�?�A��"@�nLA�|ZA͂�?P
.A��}A���A���A�|�ABe�A��@��AGj(Al��A���B�0?M%�Ag�hA\��A�p�@��A�A):�AKA? @��A�AAmw�A�@A�w�A`�3C�W�B �A��UBo�A�y/A�}�A�|NA��A�PA�|jACTA�:�A���A��A���@��}@X"1@��C��j      N                  "   1   	                     &   E         
         �                  k   5      +         
   
   /   -                                 I   2      9      5      5            %   3                        #   +                  9            &      3   #                     '   %                                 #   !            '                     !                           '                              &      /                        !   %                                 #         N��P�N���O���O��O$%TN^�}O��O��IN�s�O�P�N��NAT�O��N�W�M�?O��TO���Ng��N���N��dN��N+��O�*�N-��O;7�O^�O�H�M�
RPC��O���N�6mOAm�OxON�͓N��lN|}O�!O��%N�:�N$��N�-O��N�<O)��N�$N�sN�RSONMO�>�OxF�N(1Oh'  /  ,  �  (  �  �  k  G  ]  A  �    �  �  D  �    	      �  �    X  �    �  �  3  	  �  �     �  �  �  )  �  $  �  c  {  �  �  �  �  &  h  6  6  
�  9  �T��<D���u=P�`:�o;ě�;ě�<t�<ě�<49X<�C�<ě�<�t�<���<���<��
<�/<��<���<�/<��=o<��=ě�=+=+=t�=t�=t�=49X=L��=8Q�=]/=<j=@�=D��=T��=e`B=aG�=}�=}�=��=�+=�7L=�7L=��P=���=��
=�1=��T=�
==��>�pqtz��������xtpppppp#<Haenqof_H</#-///-)/<HSUYUUQH</--�����������������������

������A@;8?BN[gjsqge\[NBAA���	 ����������������������������������������������������������������EDEHTaz������zmaYNIE"#'/7<HJSPH=<7/(&#""|��������|||||||||| "(5BKNR^bd[NB),()012<AIKJIGA<0,,,,A>@BOPQPOBAAAAAAAAAA��
#/<DFJKH</# �,-6<Hanz������n\H</,������
	���������������� ������������������������������������������������������������������������)4=EJLJB6)
��������������������_`bn{�������{vsnhb_��������
�����������)5<BBZZIB5	��������������������������)5BNZ`fge[5)������#)5;<6)��
#*-000-#
pnot~������������vtp����������./7<HQTRH<0/........�������������������������������������������-AGJLKFB5)	���������������������������������������������������������������������������������������������������� 
#0<><8)#
 �����������������`amrz��zzma`\Z\````258@BNY[^eb_[TNB<522plt������������zttppdbdkz������������znd������ ���� $$�����������������zÇÍÓÙÓËÇ�z�r�n�l�n�p�z�z�z�z�z�z�������	��%�%�������������������������z���������������������������z�o�o�r�z�z����������������������������������Ż:�F�S�_�c�i�g�_�X�S�F�:�/�-�-�-�4�8�:�:����������������������������������������ù����������üùìéìñùùùùùùùù������'�/�3�/�'����Ϲ˹����ùϹܹ������������
��	�����������������z������Ź����������������ŽŹŲŰŷŹŹŹŹŹŹ���������������������������������������������������������������������f�s�x�y�t�s�f�c�]�a�f�f�f�f�f�f�f�f�f�f�n�{ŔŠŹ������ŹŰŠŇ�{�n�b�\�Z�[�Y�n���������������������������y�y�������������������z�~�����������y���������������y�m�`�T�E�;�9�B�M�`�m�y���5�A�N�Y�]�Z�N�A�5�(����������ƎƚƧƳƶƳƱƧƚƎƄƍƎƎƎƎƎƎƎƎ��������������������������������T�`�d�m�r�q�m�b�`�Y�T�G�=�;�:�;�G�K�T�T�	��"�.�1�0�.�+�"���	������������	�	�#�/�<�?�H�Q�H�<�7�/�#��#�#�#�#�#�#�#�#���������������������r�f�X�R�R�V�d�r����ּؼ���������������ټּּּּּֽݽ����ݽнĽ��������������������ýݾ��������ƾƾž�����������v�v����������s�}�v�u�w�o�f�Z�M�4�(����6�A�M�Z�f�s�ûлۻллû��������ûûûûûûûûû����B�O�tĂćą�y�h�[�B��������������`�y�����������������x�m�`�T�G�G�K�P�T�`������������������ۼܼ������(�5�B�N�Q�V�M�A�5�(����������	��.�;�G�T�Z�U�L�G�;�.�"��	��������	EEEEEEED�D�D�D�D�EEEEEEEEǭǡǚǔǈ�{�z�{ǇǈǔǝǡǭǵǭǭǭǭǭàâìùùùõìàÓÎÇÄÇÓØàààà��$�=�H�E�>�0������������������������<�H�U�a�i�r�t�}ÛæàÓ�z�a�1�!��!�4�<�N�Z�^�g�j�g�d�]�Z�P�N�A�>�<�5�8�A�J�N�N�A�N�Z�Z�Z�Y�N�A�>�;�A�A�A�A�A�A�A�A�A�A�����������������������������������������������
������
�����������������������������������������������������������޾s�w���������������z�s�f�b�Z�Y�Y�Z�f�s²¿������������������������¿¹²±²²������������ŹŭŠŠřŠŠŭŹ�������������"�/�2�5�/�,�"��	����������	������������������������������������������׻����ûлڻ����������ܻлĻ��������������!�(�3�3�%�!������ֺɺúɺѺ�������������������������������������������D{D�D�D�D�D�D�D�D�D�D�D{DvDoDnDkDoDpD{D{ , # -  * - E 9 g N  / [ 9 > < 5 > @ 9  : e 2 B { 1 s \ 5 +  / 6 % T A S _ F 8 6  R 5 O / ( G V N Z 4    �  l    %  *  `  ~  r  �  �  V  �  �  �  �    g  f  u  �  �    �    X  �  �  W   �  \  �  �  �  �  �  �  �  >  g  5  K  �  7  :  {        B  7  %  ;  P  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  /  !    �  �  �  �  �  �  �  g  C    �  �  N    �  r  '  j  �  f  �  �    '  +       �  |    �  /  �  w  �  �  ]  �  �  �  �  �  �  �  {  k  W  B  ,    �  �  �  z  J     �  2  j  G  �  	�  
  
�  
�    (  "    
�  
]  	�  �  �  �  �  _  �  �  �  �  �  �  �  y  d  K  ,    �  �  �  �  �  p  a  [  �  �  |  v  ~  �    i  N  ,    �  �  �  w  _  ^    �    k  b  _  b  h  m  o  h  \  L  6       �  �  f    �  I  �  )  7    �  �  �  {  G  *  ,  8  G  L  B  -    �  f  �  {  �  �  �  
    T  [  ;    �  �  �  d  "  �  ]  �  G  �  E  9  <  ?  >  ;  5  )        �  �  �  �  �  �  �  �  �  �  c  �  �  �  �  �  �  u  U  4    �  �  W  �  �  K  �  �  @  i  �  �  �  �          �  �  �  �  x  5  �  y    �  &  �  �  w  n  d  Y  O  D  8  ,  "                    �  �  �  z  f  P  8  !      �  �  �  �  \  *  �  �  !  �  D  7  *         �  �  �  �  �  �  �  �  �  �  �  �  �  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    u  k  `  V  �  �        �  �  �  g  /  �  �  ~  \  ,  �  g  "  �  �  �  �  		  	  	  �  �  �  k  -  �  �  8  �  d  �  *  @  *  �        �  �  �  �  �  �  �  �  �  �  e  =    �  �  e  #    �  �  �  �  �  �  |  g  R  7    �  �  �  �  �  P     �  �  �  �  �  �  �  �  �  �  �  �  {  e  L  .    �  �  �  �  �  �  �  �  �  �  �  �  �  o  X  ?  %    �  �  �  u  2   �    {  x  t  o  b  U  H  :  *    
  �  �  �  �  �  �  �  j  
@  [  C  �  u  �    F  W  <  �  �  )  �  b  �  
�  	E  �  &  �          	  �  �  �  �  �  �  �  �  �  �  y  l  `  U      �  �  �  �  z  ^  ;    �  �  �  �  v  :  �  �  �  V  �  �  �  �  �  �  �  �  �  �  �  �  Y  (  �  �    �  �  6  �  �  �  �  �  �  �  x  �  �  �  �  �  r  W  /  �  �    �  3  5  8  :  <  <  7  1  ,  &  #  "  !          #  &  )  ,  
�      
�  
�  
�  
�  
�  
�  
�  
i  
"  	�  	m  �  l  �  �  ^  3  v  �  �  �  �  �  �  �  t  P  "  �  �  ;  �  @  �  �  �  *  w  x  |  �  �  �  �  �  �  y  j  U  7    �  �  �  �  y  `  �  �  �  �  �     �  �  �  �  `    �  -  �  �  ^  �  �  �  �  �  �  �  p  \  F  -    �  �  �  _  +  �  �  �  O  !  !  �  �  �  r  C    �  �  t  2  �  �  R    �  W  �  �  4  �  �  �  �  a  5    �  �  s  I    �  �  �  i  =    �  �  �  )    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  ~  H  	  �  e    �  3  �  �  J  �  �  p      
  �  �  �  �  Z    �  �  '  �  �  /  �  p  �  I  �  �  �  �  �  }  d  G  !  �  �  �  O    �  �  W    �  R   �  c  [  S  L  C  :  1  )  !        �  �              �  {  f  P  6    �  �  �  �  k  G  #    �  �  �  Q    �  [  J  {  i  I  &    �  �  �  ]  0  �  �  �  m  D    �  V  6  �  �  �  �  �  �  �  �  x  k  ^  P  C  6  (      
    �  �  |  W  5    �  �  �  �  _  5  	  �  �  ~  J    �  b    I  j  }  �  s  Y  9    �  �  �  p  5  �  �  d      �  �  &  �  �  �  �  i  O  <  +      �  �  �  j  )  �  |    �  h  ^  R  F  9  )      �  �  �  �  �  �  J    �  �  m  ?  0      -    �  �  �  ~  L    �  �  E  �  �  g    �  �  5    
�  
�  
�  
�  
�  
�  
|  
O  
  	�  	�  	-  �  �  9  W  A  Y  	�  
�  
d  
F  
2  
  	�  	�  	�  	D  �  �  8  �  ,  �  �    4    9  (      �  �  �  �  �  y  [  <    �  �  �  �  �  �  �  �  �  �  S  	  �  ^    �  -  �  2  
�  	�  	5  _  w  �  b  W