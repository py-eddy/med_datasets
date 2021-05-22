CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?öE����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�^   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =��      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���R   max       @E�\(�     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����R    max       @vr=p��
     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @͗        max       @�3�          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �e`B   max       >j~�      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��H   max       B+��      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�n@   max       B+�W      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��b   max       C�h      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��<   max       C�g�      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�^   max       PQ"V      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��+I�   max       ?�%��1��      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >	7L      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�G�   max       @E�\(�     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vqG�z�     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @͗        max       @�L`          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >   max         >      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���@��   max       ?�$xF�     �  M�                     
                           /      +      1                           `   /      @            P   	   "          	                     %   	   �         N-�N2\N��O��O	m&O,rWN�i�O�uO.u O3^CN�)�N���N���N�^N@:�PW�{OkO;O~�O^�O��N��vO�mO�/�Ov O�mO�(kNF�)O蟵P��O�T�O�8�PK�N׳NoN?�OP���N�k@O�UN+�O�3�N���OI7 N�N�2�N�\O�)O"�<O�\�N;Z{O���N��N�ʉN|*�����e`B�o�ě����
��o�D���o�o�o%   :�o:�o;�o;��
;ě�<t�<#�
<49X<D��<D��<D��<D��<T��<e`B<e`B<e`B<�C�<�t�<ě�<ě�<ě�<���<�<�=o=+=\)=�P='�='�=0 �=aG�=}�=}�=�o=�O�=�\)=�hs=���=�^5=�^5=��35BNU[f[NB7533333333����������������������������������������TPPV[gt���������tgeT#+0<BILSQIH<<0.$#d][gt����������tlmgd)6BGKDB6*)����������������������������������������>8?ABGO[dhptztmh[OB>�{|�����������������#*/<BHIHGA</%#�����������������������������������������������������������������5Ntoid[B)�����������������������;::>BKN[glrttrqg[NB;���
#/8<JG@</ 
�������(042' ���������

�������!#'/<CHNTSNH</-%##!!V[`hs����������thf[Vqt���������������|tq�'7BNQSQNB5)���43;?N[gjmoolg[NMFB74������������������������������
�������������������������������������������)5BDAC>5���toosz�������������zt	
 #$#!

								����



	����������"!#/34//##""""""""���5BEDHVYL5 �����UUW^amszzzyomaUUUUUU�����{y�����������������������������������
#<Uz�zaUH</�����������������������������������������00:<INSIA<8000000000	)56962*)		78>ABKO[]][OMB777777�������������������������������������������+@;;6)��������������������������������
 
�����)5BNB?85,)NR[gqt}�tg[[NNNNNNNN��������������������ÓÖÛÖÓÌÇÅÅÄÇÉÓÓÓÓÓÓÓÓĳĸĵĳĦĚčċčĚĦĬĳĳĳĳĳĳĳĳŇŔŕŗŔŇ�{�y�{ŃŇŇŇŇŇŇŇŇŇŇ���������������������������������|���������������������������x�m�l�_�_�g�l�x�y���H�T�a�d�e�Y�T�Q�I�I�H�C�;�/��!�/�;�F�HÇÓÙØÛàÓÇ�z�z�s�z�{ÂÇÇÇÇÇÇ����*�4�6�C�D�C�?�6�*�����������������������"��������������������
��#�/�2�<�F�G�<�9�/�)�#��
�������
E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������������������������ʼϼӼʼʼ����������������������������ʹ���	���� ��������������������������H�T�a�c�d�a�T�I�H�B�H�H�H�H�H�H�H�H�H�H���������ƿ��������m�`�;�/�'�/�9�G�m���������������ĿɿͿȿĿ��������������������B�[�h�t�zāĂā�z�t�h�[�O�B�6�-�,�.�3�B�/�;�G�F�@�5�,�"���	�������	����"�/�/�;�T�a�t�z�v�m�a�T�H�/��
�������	��/�������Ⱦ���������������������������������	�����������������������������M�f�������������s�f�M�D�4�-�*�*�4�C�M�;�E�K�G�F�A�;�8�.�"��	���� �"�.�5�;Ƨƫ����ƳƭƠƚƎƁ�{�z�t�k�m�uƈƎƢƧ�������������������ƽƱƲƳ��������ÇÓàåáà×ÓÉÇ�~ÂÇÇÇÇÇÇÇÇ�������(�A�j�|�|�q�Z�8�(��������ùϹ����%�'�%�����蹶������������čĚĳ��������������ĿĳĦčā�v�s�zāč��<�I�X�^�a�h�j�a�]�U�<�0�#�������x���������л������ܻл��������w�t�x�y�{�}�y�t�m�d�`�\�`�m�o�y�y�y�y�y�y�y�y�"�$�.�3�/�.�"�����!�"�"�"�"�"�"�"�"�	��"�.�3�6�.�"���	�	�	�	�	�	�	�	�	�	¿������ ��
�����g�V�H�N�[�t¯¿���������������������������������������̾����&�4�A�Z�s�z�z�n�f�X�M�4�'����������������������������5�=�>�2�+��������ֿпοѿݿ����5��(�5�8�A�N�W�Z�]�Z�N�A�5�4�(� �������(�4�A�M�P�_�f�k�f�Z�A�4���������������ۺֺ˺ֺ���������r�~�����������������������~�}�r�o�i�r�r�������ɺкɺǺ�������������������������ÇÓàìï÷ùùùîìàÓÍÇÀ�}�~ÇÇ�����ʼּ��������ּʼ���������������!�F�{�������x�_�:�-�������������y�����������y�l�l�l�m�q�y�y�y�y�y�y�y�yD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D|D~D�D�D������������������������}�s�q�q�s������������������������������������������������EuE�E�E�E�E�E�E�E�E�E�EyEuErEuEuEuEuEuEu u X D * 4 _ O B L & 9 & � J E ` -   v 8 ] % : S 8 3 - � - * F 6 8 U P @ 0 ( ; j r ` X B E ' < L < - O W o    �  n  /  0  )  �  �  :  �  v    �  �  9  X  3  �  �    J  �  :  M  R  �  0  [    c      X  C  O  r  w  �  �  >  �  �    Q    �  )  i  �  c  �  &  �  ��e`B�#�
���
<�t�%   ;o;�`B;D��<u<t�<�o;��
;D��<t�<D��=Y�<�j=T��<�t�=y�#<���=��<�/<�1<�1<�`B<���=��=�l�=�hs=49X=�9X<�=+=t�=�S�=,1=�C�=,1=��=H�9=��=q��=���=�O�=��T=���=�"�=���>j~�=��=��>	7LBB��B/�B	i!B&mB
)�B��B��B�B-�BQ�BT�B"�bB #[B��B��BɚB��B{�BPyB�nB��BTtB�B�#B�tB��B"SB7BRB�0BƌB8RB]�B�B��A��HB��B `�B��B-CB��B&bYB��B}B![�BsB�	B+��B�B�oB	(B;�B��B�
B9zB	I�B%��B
L�B��B�B0FBA�BAB�IB"��B ?�BBSB1�B4PB�7B�gB@>B�{B�BH�B��B@
B��B�B#��B?�B>�B�"B�LB=BE�B�;B?OA�n@B��B ��B@�B�qB�wB&z8B��B�B!B�B?�B��B+�WB@9Bq�B	ZhB�wA��QA�0IA��A�[�@�]�A��A���A��AЗHA�3wC�hA��@�V�?D��A��BAm/�As��A�KA�6�A��AL.EA�r�AA2A^��BB]�A�P�A8m�>��bA෮A��@�sgAkO�A_#ZA^nA���A��A:��@�UA�A��A9@D�@��@&+&A��A %�@zgsArUC���A��lA�P�C��qA�x�A�V=A��A�~�@��_A��%AɁ A��A�z�A���C�g�A���@��}?P �A�=Ah� Asj:A�;A�x�A�|�AKA�|5AA(�A^�iB@5Bt�Aɦ�A=�>��<A�|�A�`@@�cAj�GA`n�A^��A��hA�rA:\@��wA��BA�j�A:��@D]@E�@$�A�}0A m�@{0TAt�C�ڍA�`�A�~C�	k                                                0      +      2            	               `   0      A            P   
   "      !   	                     &   	   �                                                         5            %                        1   '   '   #   %            7      "      +                        -                                                               3                                    1         #               1      "      +                        %               N-�N2\N��Nga�O	m&O��N�i�O�uN6� O� N�MyN���N���N�^N@:�PA��OQs�OAY8O^�O�#�N��vN���OU��N~�'O�mO\��NF�)O蟵O~rpO���O�qO��N׳NoN?�OPQ"VN�d�O�UN+�O�3�N���OI7 N�N�2�N�\O�)N�rvO�?�N;Z{O&{N��N�ʉN|*�  :  �  �  L    y  �  
  Z  {  }  �  �  �  �  �  J  ]  K  w      �  �  �  �  )  n  
z  �  �  �  b  �  ~  �  �  J  �  �  g  w  X    {  �  {  �  7  '  �  �  �����e`B�o<o���
�D���D���o;ě�%   ;�o:�o:�o;�o;��
<t�<#�
<�C�<49X<���<D��<u<u<�o<e`B<�o<e`B<�C�=]/<�h<���=��<���<�<�=8Q�=C�=\)=�P='�='�=0 �=aG�=}�=}�=�o=��P=���=�hs>	7L=�^5=�^5=��35BNU[f[NB7533333333����������������������������������������Y[^gtu~~tgb[YYYYYYYY#+0<BILSQIH<<0.$#agt���������xtnohga)6BGKDB6*)����������������������������������������?BEELO[_hmtvtrh[OCB?��������������������#*/<BHIHGA</%#����������������������������������������������������������������5Njodaa[B)�����������������������?>=>BCN[gnqqohg[NKB?���
#/8<JG@</ 
���������#((%�������

�������'%#+/<@HLSQHH<0/''''Y[`acht��������the[Y���������������������'7BNQSQNB5)���=96@BEN[ghlmmjg[PND=������������������������������
����������������������������������

����������)5B??B5���}�����������������	
 #$#!

								����



	����������"!#/34//##""""""""����4==OPL5����ZVX`amrywnmaZZZZZZZZ�����{y�����������������������������������
#<Uz�zaUH</�����������������������������������������00:<INSIA<8000000000	)56962*)		78>ABKO[]][OMB777777����������������������������������������������)563&������������������������������

������)5BNB?85,)NR[gqt}�tg[[NNNNNNNN��������������������ÓÖÛÖÓÌÇÅÅÄÇÉÓÓÓÓÓÓÓÓĳĸĵĳĦĚčċčĚĦĬĳĳĳĳĳĳĳĳŇŔŕŗŔŇ�{�y�{ŃŇŇŇŇŇŇŇŇŇŇ�������������������������������������������������������������x�m�l�_�_�g�l�x�y���T�`�`�T�S�O�H�F�;�/�-�"�!�"�#�/�;�H�J�TÇÓÙØÛàÓÇ�z�z�s�z�{ÂÇÇÇÇÇÇ����*�4�6�C�D�C�?�6�*�����������������������������������������������������#�/�<�@�@�<�5�/�$�#���
���
��E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������������������������ʼϼӼʼʼ����������������������������ʹ���	���� ��������������������������H�T�a�c�d�a�T�I�H�B�H�H�H�H�H�H�H�H�H�H�������¿����������y�`�T�;�2�)�1�G�m���������������ĿĿʿǿĿ��������������������B�O�[�h�s�t�|�{�u�h�[�O�B�6�5�2�4�6�;�B�/�;�G�F�@�5�,�"���	�������	����"�/�"�/�;�H�T�a�h�m�l�i�a�T�H�;�,�����"�������Ⱦ����������������������������������������������������������������M�X�f�s�����������s�f�Z�M�?�@�=�A�I�M��"�.�8�6�.�&�"���
���������Ƨƫ����ƳƭƠƚƎƁ�{�z�t�k�m�uƈƎƢƧ������������������������Ʒ��������ÇÓàåáà×ÓÉÇ�~ÂÇÇÇÇÇÇÇÇ�������(�A�j�|�|�q�Z�8�(��������Ϲܹ�������������ܹϹù������ù�ĚĦĳľ����������ĿĳĦĚčĄ�z�w�~ĈĚ��#�<�I�U�]�`�f�h�^�X�I�<�-����	���������лܻ����ܻлû����������������y�{�}�y�t�m�d�`�\�`�m�o�y�y�y�y�y�y�y�y�"�$�.�3�/�.�"�����!�"�"�"�"�"�"�"�"�	��"�.�3�6�.�"���	�	�	�	�	�	�	�	�	�	¿����������
���t�]�T�U�d�t¥¿���������������������������������������̾����&�4�A�Z�s�z�z�n�f�X�M�4�'����������������������������5�=�>�2�+��������ֿпοѿݿ����5��(�5�8�A�N�W�Z�]�Z�N�A�5�4�(� �������(�4�A�M�P�_�f�k�f�Z�A�4���������������ۺֺ˺ֺ���������r�~�����������������������~�}�r�o�i�r�r�������ɺкɺǺ�������������������������ÇÓàìï÷ùùùîìàÓÍÇÀ�}�~ÇÇ���ʼͼּ�������ּʼ����������������!�6�F�_�u�x�{�l�[�F�:�-�!�� ������y�����������y�l�l�l�m�q�y�y�y�y�y�y�y�yD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������������������������}�s�q�q�s������������������������������������������������EuE�E�E�E�E�E�E�E�E�E�EyEuErEuEuEuEuEuEu u X D 8 4 Q O B : # 8 & � J E _ *  v , ] % 7 4 8 2 - � #  D ( 8 U P G , ( ; j r ` X B E ' ' E <  O W o    �  n  /  }  )  2  �  :  N  .  �  �  �  9  X  �  �  �    C  �     �  y  �  �  [    �  �  �  s  C  O  r  �  �  �  >  �  �    Q    �  )    �  c  ]  &  �  �  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  :  V  q  �  �  �  �  �  �  �  �  �  u  g  Y  J  :  *      �  �  �  �  �  �  �  �  m  W  B  ,       �   �   �   �   �   y  �  �  �  �  �  �  �  �  �  �  �  x  m  a  S  F  8  +      �  �  �  �    (  /  /  /  1  >  K  J  5    �  g     w  �         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  s  v  x  x  w  v  u  t  s  q  m  j  e  _  Y  P  ?  /    �  �  �  �  �  �  �  �  �  �  u  T  +  �  �  �  d  .   �   �  
         �  �  �  �  �  �  �  �  �  �  �  �  �  �    p  j  ~  �  �  �  �  &  @  P  Z  W  I  <  -    �  �  �  i    m  t  y  z  {  y  o  c  S  C  2      �  �  �  �  V      �  :  W  g  q  y  |  v  s  f  Z  T  N  =  $  �  �  �  h    �  �  �  �  �  �  �  �  �  ~  s  g  [  N  A  5  -  (  "      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  m  W  @  )    �  �  �  �  �    6  �  �  �  �  �  �  �  �  �  n  R  8     	  �  �  �  �  �  l  �  �  �  �  �  r  H    �  �  �  &    �  �  3  �  <  _   �  @  G  G  A  5  '      �  �  �  �  g  6     �  �  �  �  �  C  T  Z  \  W  K  ;  .      �  �  �  e    �  �  c  �  �  K  G  D  @  ;  7  0  (       
  �  �  �  �  �  �  �  �  �  �  !  A  [  p  w  v  r  h  W  :    �  r    �  9  p  �  b        �  �  �  �  �  �  x  ^  D  %    �  �  j  3   �   �  �  �        �  �  �  �  �  �  d  3  �  �  d    �  !  �  �  �  �  �  �  �  �  �  ~  e  K  1    �  �  �  �  Q     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  M  1     �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  c  ]  V  L  @  3  �  �  �  �  �  �  �  �  �  �  �  �  b  B     �  �  �  ]    )        �  �  �  �  �  �  �  �  �  r  X  >  '    �  �  n  _  H  0  $          �  �  �  �  �  �  v  B  
  �  Q  	  	�  	�  
  
.  
T  
n  
y  
p  
I  
  	�  	I  �  h  �  \  ;  �  �  O  �  �  �  �  u  Q  &  �  �  �  [  9    �  �  A     �  �  �  �  �  �  �  �  �  �  �  w  d  P  8      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  L  �  �    �  �  )  M  b  Q  ?  .    
  �  �  �  �  �  u  W  9    �  �  a  !   �  �  �  �  �  �  �  �  �  �  �  �  �  {  u  l  c  Z  Q  H  ?  ~  u  l  d  Z  Q  H  2    �  �  �  �  �  c  E  '     �   �  5  �  �  �  �  �  �  �  �  �  j  0  �  �  g  
    �  z  k  �  �  �  �  �  �  �  �  �  s  a  O  <  *      �  �    9  J  '      �  �  �  �  �  �  �  X  '  �  �  �  ,  �    �  �  �  �  �  �  �  �  �  �  �  �  x  i  W  D  2  .  .  -  -  �  �  �  o  P  0  d  M  -    �  �  �  B  �  �  B  �  �  �  g  O  8      �  �  �  �  o  X  K  ;    �  )  s  �  �  i  w  r  f  W  C  *    �  �  �  �  �  g  0  �  �  �  J  �    X  U  R  N  K  G  C  >  9  5  .  '        �  �  �  �  �      �  �  �  �  �  �  r  W  =  %        �  �  �  F  �  {  z  y  v  r  n  i  c  ^  U  K  A  8  0  ,  0  4  9  ?  F  �  �  {  ^  =    �  �  �  h  0  �  �  J  �  �  I  �  �  N  3  N  g  y  m  S  5    �  �  �  T    �  U  �  b  �  E  �  t  �  �  �  �  �  �  v  ?  �  �  m  &  �  �  /  �  :  >   �  7  2  .  *  '  %  %  %  (  +  .  1  4  8  <  B  I  S  `  n  o    |  �  k  �    %  %    �  p  �    &  �       �  �  �  �  �  �  y  h  P  8      �  �  �  o  2  �  �  }  T  4  �  �  �  �  �  �  o  U  5    �  �  �  ^  5    �  �  S  �  �  �  s  ^  H  .    �  �  �  �  �  ~  h  L  -  �  �  �  Y