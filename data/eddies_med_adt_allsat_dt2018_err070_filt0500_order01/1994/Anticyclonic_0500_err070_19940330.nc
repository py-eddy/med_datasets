CDF       
      obs    4   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?� ě��T      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�\�   max       P�      �  |   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       =��T      �  L   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @F�Q��            effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @vS
=p��        (<   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P            h  0\   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @�s�          �  0�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���
   max       >5?}      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�@   max       B42i      �  2d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�]�   max       B47�      �  34   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?M�W   max       C�^      �  4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?P��   max       C�W�      �  4�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max                �  5�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M      �  6t   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  7D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�\�   max       P`7S      �  8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���$tS�   max       ?���҈�      �  8�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �o   max       =��      �  9�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @F�Q��        :�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�{    max       @vR�\(��        B�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P            h  J�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @��           �  K,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?   max         ?      �  K�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?x*�0��   max       ?��u%F     @  L�               =      
   [               	                        Y      "   5         =         G   
                  
         "   	   ~         
             	   %      `O 5�Nc�@M��FO�y�P�Nb�NL7�P�ˉN�NZ��N�N���N�I�O��{NF-�N���M�\�NN�O�'O�i:Pp�bN7�PJO}Oq�Pm݋O���O��O�P�N�9Of��N�BO/�ZN4�GN���O_=�O�*kNi=�O���N��P
J�N��N��yN�]yO��
N�O<]�N���O��ON�O��-�o�o�o�ě���o�D���o:�o;o;ě�;ě�;ě�;ě�<o<o<49X<49X<D��<T��<e`B<e`B<e`B<e`B<�o<�o<�C�<�C�<�C�<�t�<���<��
<�1<�1<�9X<�9X<ě�<�=\)=\)=��=��=�w=#�
='�=,1=0 �=0 �=0 �=8Q�=]/=]/=��TZ[\`agt�������xtmg`ZHHN[]gllg[NNHHHHHHHH��������������������)5BGPTXNHB)������6akeB��������C??HKUY\YUMHCCCCCCCCb\bhnt����thbbbbbbbbk|�������������uk�������������������))575-)3/25BGGBB53333333333������������������������������������������������������������4246BHMGB64444444444����������������������������������������������������������������������
�����";HLbedaSHD;/"�����)?@;0)��������������������������
$',1<U^maYaU>/�#(/<HPTUTQH</#/)((/7<?HQU[XUOH<///rqzwm~�������������r����
/@LUUTMH/#
�~xvx~~�������������~��������������������qjt����������������q��������������������������������)-121)$)25:???=5)8;>BOXQOJB8888888888�������������������������������������/9974,#������AHHUaekhaUSHAAAAAAAA3?BO[YJD:3.)���������������������������*/&�����zz������������zzzzz���������������������{}|~������������������������������������������������������������&(&	�������� ���������������������������������������������������

������zÀÇÓàêììàÓÇ�z�n�l�a�^�a�f�n�z���������������������������������������ſ����Ŀƿ̿Ŀ����������������������������6�B�O�Q�[�h�t�~Ā�y�t�h�[�O�I�B�>�)�&�6�T����z�a�`�^�/�	���������������"�;�H�T���������������������������������������������������������������������������������B�[�g�z�y�|�t�[�5�)����������������)�B���)�6�7�6�)�%�!�������������Z�f�h�g�f�f�\�Z�V�M�J�M�M�W�Z�Z�Z�Z�Z�Z��(�4�9�9�4�(�������������������������������������������ú�������Ź�����������������������������~���������������������������~�w�r�t�|�~�l�y���������y�l�`�f�l�l�l�l�l�l�l�l�l�l�)�6�:�:�6�6�)� ������"�)�)�)�)�)�)������������������������b�o�y�{�ǀ�{�o�j�b�[�`�b�b�b�b�b�b�b�b�(�4�A�M�Z�h�t�}�����s�f�Z�M�(����(�/�;�T�a�m�y�w�m�T�H�;�5�"���	����/�	�"�T�u������s�T�H�"�����������������	�)�6�9�@�6�*�)������"�)�)�)�)�)�)�)��(�2�Z�f���������f�M�4���������EEEE&E*E2E7E1E*EEED�D�D�D�D�D�D�E��������������������������������޾A�Z�f������þ��������s�M����	��(�A�`�m�y�����������y�m�T�G�;�8�1�0�4�<�G�`�M�Z�f�����������������s�Z�M�A�;�9�?�M����*�6�;�C�D�J�C�6�*�����������������4�M�Y�f�p�j�M�4���ۻڻ�ܻ����"�'�3�7�3�3�'��������������y�����������н������нĽ������������y�����	���	�����ؾܾ���������)�.�5�B�N�[�^�g�k�g�e�[�N�B�5�)� ���)�ܻ����� �����ܻԻܻܻܻܻܻܻܻܻܻ�àâìùû��ùöìàÛÜÚÔàààààà���� ������"�!������ݿѿӿݿ�����(�5�=�:�;�5�(����������������A�N�N�S�R�N�A�5�2�4�5�9�A�A�A�A�A�A�A�A�e�r���������ɺҺº������~�e�Y�L�E�M�\�e�y�����������������y�q�n�l�b�l�u�y�y�y�y�������ּ�����ּʼ������������������
�
�����"���
�����������
�
�
�
�/�<�H�T�U�H�B�<�/�)�%�*�/�/�/�/�/�/�/�/���������Ľнݽ�ݽ׽нĽ���������������������������ûÛÓÇ�z�o�c�q�zÇàôù�������������������������������������������
��#�0�<�I�T�V�U�S�I�<�0�(����
��
�a�n�y�t�p�n�a�U�H�F�H�R�U�]�a�a�a�a�a�a�4�@�K�b�Z�M�@�4�*������ܻ���� �'�4���ûȻǻû����������x�l�^�U�_�e�x������D{D�D�D�D�D�D�D�D�D�D�D�D�D{DvDiDiDkDrD{ 7 D N - S G Z D n Z 2 d 4 3 > U P 2 B #  � O   8 ( : 9 O I x Q 8 ` N 6 D N b L > x 3 K  l A Q } o   _  y  "    w  �  o  �  �  �  5  �  �  9  Q  �    d  Z  �  �  w  �  �    �  �  b  O  �      �  ~  _  �  �  �  �  �  �  �  �  �  '  �  n  �  �  �    ���
�D���ě�;�`B=e`B:�o;�`B=�^5;��
<#�
<o<���<u<�`B<e`B<�C�<T��<�t�=0 �=+=��<ě�=@�=�C�='�=<j=��w=8Q�<�`B=�E�<�h=D��<�`B=o<�h=o=�w=��=�w=�hs=<j>(��=@�=e`B=P�`=��P=<j=e`B=Y�=�^5=��P>5?}B	��B�Bo?B�B�)BiSB
B (B4�Bc�ByB�&B�BB�B�8B42iB8�B"�$A�@B^FB�BB�-BxfB�B��B�.B��B@�B!
GB]PB x�B�)B��B]B!��B�]B�1B�IBc�B,rOBcB6�B�B)f�Bp�B@�Bl�B�dB�B�LBжB	��B�hBC�B@BA4BB%B�RB�YBC�B��B`B�B��B(�B;�B�#B47�B=�B"��A�]�B��B��B��B?�B�UB��B8`B�kBB!@BA�B ?�BH6B�B<�B!��B��B��B�B:�B,@�B��B?�B�FB)RB�TBHB�B�?B?�B�B�A�s�A�Z;AwYA�ϦA��nA�E@��LA� VAՁ�A?G�A7'�A��?M�W@��A�A֕'AW�B�\A;�9A���A��4A�7+A;�AC�^A�/<AA`BAiV�ACŦA���@Ŭ?{��A%��AW�YA�z@�G�A�%SA���A�ǘA���@
�A&�@�G�A�AÁsA%$�A���A�)aA��Aƺ�@��{@�N�C��*A�VA��Av��Aڀ!A�i�A��c@��A�UA��+A?oA6�lA��?P��@�A�AւBAW�B�|A;�A�|�A���A�M�A<!�C�W�A�{0ACAi�AD�aA��@�@�?S�2A'=AW>BA��h@�I�A�~�A���A�|�A�}�@�{A�K@��PA���A�u�A#��A�DA��A�'A�}�@��i@��FC��               =      
   [               
                        Z      "   5         >         G   
               	   
         "   	            
             	   &      a               M         9                                    !   /      +         5   !         /                              #      )            #            '                     1                                                !      !         5                                                                              O 5�Nc�@M��FO�y�PM�Nb�NL7�O�GmN�NZ��N�NN!�N��Os�NF-�N?��M�\�NN�OA_�O�H/O�F�N7�O���Og�7N�t�P`7SO{p�N��N�^_O���N�9OX�N�BO/�ZN4�GNYAUO_=�O�<#Ni=�O&[N��O�$�N��N��yN�]yOy�$N�O)H�N5.�OZ��N���OX��  4  �  q  p    �    �  E  +  Z  �  �  s    �  L  =  �  �  �    �  	i  �  �  |  Q  �  �  #  \  >  "  �  �  �  Z  W  k  �  �  l  �    �  �  �  �  ~  E  ��o�o�o�ě�<u�D���o=T��;o;ě�;ě�<t�;�`B<u<o<D��<49X<D��<��
<�o=H�9<e`B<���<���<ě�<�t�=�w<�<���=,1<��
<�9X<�1<�9X<�9X<���<�=#�
=\)=H�9=�w=���=#�
='�=,1=<j=0 �=49X=@�=u=u=��Z[\`agt�������xtmg`ZHHN[]gllg[NNHHHHHHHH��������������������)5BGPTXNHB)������)6CORJ6����C??HKUY\YUMHCCCCCCCCb\bhnt����thbbbbbbbb���������������������������������������))575-)3/25BGGBB53333333333������������������������������������������������������������4246BHMGB64444444444�����������������������������������������������������������������������������"/;H_cbQHA;/"�����)//-)�������������������������#+16<HU[`]UNM@/#*/<HLRSROF</#../2<HHSOH@<9/......r{|o��������������tr
#/<@GKKHF</#
������������������������������������������������������������������������������������������)-121)$)25:???=5)8;>BOXQOJB8888888888���������������������������������������
#/23/'����AHHUaekhaUSHAAAAAAAA$)6BFAA?;63)��������������������������������zz������������zzzzz���������������������{}|~������������������������������������������������������������$'$�������� ���������������������������������������������������

	������zÀÇÓàêììàÓÇ�z�n�l�a�^�a�f�n�z���������������������������������������ſ����Ŀƿ̿Ŀ����������������������������6�B�O�Q�[�h�t�~Ā�y�t�h�[�O�I�B�>�)�&�6�	��/�6�;�I�G�I�C�=�/��	�������������	���������������������������������������������������������������������������������)�5�B�G�S�W�P�B�5�)��������������)���)�6�7�6�)�%�!�������������Z�f�h�g�f�f�\�Z�V�M�J�M�M�W�Z�Z�Z�Z�Z�Z��(�4�9�9�4�(����������������������������������������������������ҹ��������
�������������������������������������������������~�|�~��������l�y���������y�l�`�f�l�l�l�l�l�l�l�l�l�l�)�6�9�9�6�3�)�����$�)�)�)�)�)�)�)�)������������������������b�o�y�{�ǀ�{�o�j�b�[�`�b�b�b�b�b�b�b�b�4�A�C�M�Z�^�f�j�r�r�f�Z�M�?�(���"�(�4�/�;�H�T�a�j�q�u�o�T�H�;�"����
���/�/�;�H�W�e�h�f�b�U�H�/�"������� �	�"�/�)�6�9�@�6�*�)������"�)�)�)�)�)�)�)��(�4�I�Z�s������|�s�f�Z�M�4���
��EEEE%E*E0E4E.E*EEED�D�D�D�D�D�D�E������������������������������������Z�f�s���������������q�M�%�����(�A�Z�`�m�y���������~�y�m�`�T�J�G�@�=�>�E�S�`�s�|�����������������s�f�e�Z�V�U�Z�f�s���*�6�8�A�C�6�*��������������4�<�@�H�M�K�@�4�'��������� �����"�'�3�7�3�3�'����������������нݽ������нĽ����������z���������������	���	�����ؾܾ���������)�.�5�B�N�[�^�g�k�g�e�[�N�B�5�)� ���)�ܻ����� �����ܻԻܻܻܻܻܻܻܻܻܻ�àìùú��ùôìàÝÞÝàààààààà���� ������"�!������ݿѿӿݿ������(�0�5�9�8�5�4�-�(�����������A�N�N�S�R�N�A�5�2�4�5�9�A�A�A�A�A�A�A�A�e�r�~���������������������~�r�g�e�Y�[�e�y�������������y�t�q�l�h�l�v�y�y�y�y�y�y���������ʼܼ��޼ּʼ������������������
�
�����"���
�����������
�
�
�
�/�<�H�T�U�H�B�<�/�)�%�*�/�/�/�/�/�/�/�/���������Ľнݽ�ݽ׽нĽ�����������������������������øØÇ�z�s�g�s�zÇàöù��������������������������������������������#�0�<�H�I�S�T�Q�I�<�0�)�#�������a�n�q�q�n�n�a�U�R�U�V�a�a�a�a�a�a�a�a�a�M�W�S�M�@�5�'�������������%�'�@�M���������������x�t�l�_�\�_�l�x����������D{D�D�D�D�D�D�D�D�D�D�D�D�D|D{DnDnDoDqD{ 7 D N - N G Z < n Z 2 < 9 & > M P 2 9    � ]   5 ) " 7 % I t Q 8 ` M 6 B N I T & x 3 K ~ l > L y X   _  y  "    �  �  o  5  �  �  5  l  �  :  Q  c    d  �  �  �  w  �  �  �  �  �      /      �  ~  _  �  �  Q  �  p  �  A  �  �  '  �  n  o  X  C  �  �  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  4  .  '        �  �  �  �  �  �  o  ^  T  H  8    �  �  �  �  �  �  �  �  �  �  �  �  �  {  o  _  O  >  .      �  q  k  d  ^  W  Q  J  D  >  7  '     �   �   �   �   �   o   U   ;  p  h  b  ^  Y  T  M  F  =  2          �  �  �  �  �  j  �    �  �  �        �          �  �  f    �     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  d        �  �  �  �  �  �  n  M    �    ?  �  �  q  '   �  +  �  �  7  x  �    ]  �  �  �  �  �  �  B  �  R    �  �  E  <  4  +  #      
    �  �  �  �  �  �  �  �  �  �  �  +  !        �  �  �  �  �  �  {  e  O  9  "  
   �   �   �  Z  V  Q  M  H  D  ?  ;  7  2  +  #         �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  P  $  �  �  ~  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  m  G    �  �  l  .  �  #  @  Y  j  q  s  s  o  i  ]  J  .    �  �  �  I    �      �  �  �  �  �  �  �  �  �  �  �  t  Z  @  %  	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  U    �  �  7   �  L  H  D  ?  ;  7  3  /  +  &  #                    =  6  0  )  $        �  �  �  �  �  {  Y  6    �  �  R  V  �  �  �  �  �  �  �  �  w  T  .    �  �  K  �  [  �   �  �  �  �  �  �  �  �  f  6    �  �  �  �  v  M    �  �  �  �  $  �  �  E  r  �  �  �    ]  ,  �  }    �  �  �  �      2  f  �                
       �  �  �  �  �  �    @  Z  l  z  �  �  �  z  f  P  6    �  �  �  Y  �  �    	e  	i  	f  	Y  	B  	&  	  �  �  |  8  �  �  1  �  *  c  o  c  E  ;  b  �  �  �  �  �  �  �  �  h  F  "  �  �  �  l  >    �  �  �  �  �  �  �  �  �  �  �  �  z  i  c  S  /    �  �  4  B  �  �    K  j  y  |  r  N    �  �     �    S  E  �  u  �  �  �  �    !  6  E  O  P  G  .    �  �  I  �  N  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  e  [  Q    �    L  �  �  �  �  �  �  O    �  �  j    �  �  �  w  #        
  �  �  �  �  �  �  �  i  8  �  �  |  9   �   �  Z  X  O  E  8  *      �  �  �  �  �    �  5  �    \   �  >  ;  8  5  /  *  !        �  �  �  �  �  �  �  �  {  i  "          �  �  �  �  �  �  �  �  �  �  �  �  �  N    �  �  �  �  �  �  �  �  l  U  =    �  ~  A    �  �  {  5  �  �  �  �  �  �  �  �  �  v  0  �  K  C  ;    �  �  �  S  �  �  �  �  �  �  �  �  v  ]  A  #    �  �  �  }  Y  6    L  T  X  Y  W  M  =  &    �  �  w  B    �  �  <  �  Q  �  W  K  ?  3  '         �  �  �  �  �  �  �  �  �  v  j  ]  5  T  Y  Y  a  f  k  i  c  d  H  !  �  �  �  :  �    0  E  �  �  �  �  �  �  �  �  �  �  v  c  P  :  %    �    8  k  �  �  &  �  �  �  �  �  �  �  I  �  s  �  l  
�  	�  4  �  �  l  \  L  ;  *      �  �  �  �  �  �  }  U  1    �  �  �  �  �  �  �  �  �  �  �  �  �  g  5    �  �  P    �  W  �    �  �  �  �  �  �  �  o  W  ?  (    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Z  %  �  �  >  �  |    �  /  �  �  �  �  �  �  �  �  �  �  �  �  q  `  O  :      �  �  �  �  �  �  �  �  �  �  �  �  u  ]  @  !    �  �  �  �  w  �  �  n  z  �  �  �  �  �  �    p  Z  >     �  �  �  �  g  <      |  {  }  t  X  0    �  �  �  �  e  ,  �    v    T  �  �  �  �  *       ?  *    �  �  �  h  /  �  �  R  �  �  N  ,  Y  y  �  o  Y  =    �  �  D  �  <  z  �  �  
�  	�  �  6