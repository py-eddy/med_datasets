CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�I�^5@      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Na�   max       P]}z      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��t�   max       =��m      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F�z�H     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(��    max       @vw�z�H     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @O�           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ϥ        max       @�`          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �49X   max       >1&�      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B/�"      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B/�{      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C�q�      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�%�   max       C�\w      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Na�   max       P�[      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�[W>�6z   max       ?�'�0      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       >+      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F�z�H     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @vw�z�H     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @O�           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?uL�_��   max       ?��2�W�     p  S(   
                  #            �      	      E   $                     	         -                  C          �      G   
         <   /                                       ;             -N�}�N��O�`O���O�O�N2��PFTO)��P1��N(ǟPלN(�3NA�NȹP]}zO�oN1��O��ENEIN�_�O��lO���O5,�O��-Oe~�P�^N+�[Nf��N�(O%b	N���P+M�N�<�O���O�W�N���PHL�N�ޡNHÆNa�O���P�)O	 �N�xVOɓ�N�h�O8c�N3�.OHd�OG*|OF{�O#�GN2"�O<�2O�FOOy��N�n�N�O����t���C���o�u�D���ě���o��o��o%   :�o;ě�;�`B<#�
<#�
<D��<D��<e`B<�o<�o<���<��
<��
<�1<�9X<�9X<�j<���<���<�/<�h<�h<�=o=o=C�=C�=C�=C�=t�=�P=��=��=�w=#�
=#�
='�=,1=<j=]/=]/=u=y�#=y�#=��=��P=��
=��
=��m����������������������������������������khhmst�����������tkk68CLWanz������znUH=6QPRanu�����������znQ  it����������������ti������������������������������������������������������������:9?J[gt���������gNB:$)6BFIB;61+)$$$$$$$$���������������������������-5=:)��������5BPRMJD4/)���������������������3>N[mg^fVNB5)KOS[hooh[OKKKKKKKKKKw�������������������������������������������
#/4585/#����������������������gkt��������������wkg"/:;>HILLJ?;/"

�
#4>K`ff_I0#
������������������������������������������������������������MKSZ[bht}��|wtrh[TOM{y|���������������{{�����
#,-'������� $%  425BCN[n~���}tg[NIA4����

��������������������������������$)5BILQRRKB5����������������������759<HNSRH<7777777777;<HU[\UHG<;;;;;;;;;;-+/2;HOTaehmmhaTH;1-����������������������������������������g`\\bhltx{}tqhgggggg���)5?C?5-)���� �
%&#      #/=CILJA=<3/,&������������������)56;>@>:5) ��������������������������������������������������66**.1666666666quz���������������zqLJGGKT[hty|}|��th[OL��������

��������������������������������������������������+..)�����=�I�V�b�o�u�o�b�`�V�K�I�C�=�3�0�-�0�9�=���������������������������������n�zÇÓààãàÝÓÍÇ�z�x�o�n�j�f�n�n���������������������������������������E�E�E�FF	FFFFE�E�E�E�E�E�E�E�E�E�E��������������s�p�s�u�������������������������������������������������������������L�Y�`�]�Y�R�L�3�'�������'�3�@�G�L�N�g�s�������������s�g�Z�A�5�(�#�$�,�7�N�����������������������������������������)�6�hāċĈ��m�U�E�7�6��������)����������������������������������������Ó×ÖßÓÇ��z�u�zÇÌÓÓÓÓÓÓÓÓ�����������������������������������������	�/�;�H�L�a�z���z�m�T�H�3�(��	�������	�G�T�`�m�u�{�s�m�m�`�T�;�.��	��������G�O�U�[�S�O�B�6�4�6�9�B�H�O�O�O�O�O�O�O�O�����(�A�N�W�U�N�L�A�5�����ۿ߿���H�K�U�]�a�U�H�?�>�D�H�H�H�H�H�H�H�H�H�H�/�<�B�H�O�L�H�A�<�7�/�&�#���!�#�-�/�/��(�5�A�N�[�f�f�e�Z�N�A�(���
�����G�J�T�`�m�y�����������y�m�`�T�B�<�;�C�G��"�'�.�;�>�=�A�A�;�9�.�"��	����	�����������������������v�s�n�o�s�y�������"�/�;�G�P�T�]�T�N�H�;�/�"�	�����	������ּ�� �����ʼ�������h�X�X�i����ûλлѻлʻûûû����������ûûûûûû���!�+�,�!��
��������������ìù��������ùðìàÓÇÃÅÇÓàâìì�-�:�F�S�V�_�e�e�_�S�F�:�-�$�!���!�)�-�"�/�;�H�H�M�H�C�;�3�/�"�������"�"���	�"�-�3�1�-�%��	���������������������.�6�;�A�;�4�.�"��	�������	�
��"�&�.�.�g�~�t�g�[�N�B�5�)�"�!�%�)�5�gD�D�D�D�D�D�D�D�D�D�D�D{DwD{D�D�D�D�D�D���*�6�?�C�K�O�C�6�.�*�(���������\ƚƳ���������
������ƳƚƁ�u�l�V�S�\�����Ŀ̿ѿݿ����ݿҿĿ�����������������������������������������������������ùûû��ùìãéìøùùùùùùùùùù������#�;�G�<�8� ��
�����������������������)�6�@�O�B�7�3��������øóú�����h�tāčĕĚĜěĚčĈā��t�h�d�[�T�c�h�������������������������~�������<�I�X�b�i�j�b�U�I�0��
���
����#�0�<���������������������������������������ؾ�����������s�f�Z�M�A�9�A�C�M�Z�f�s����<�=�<�7�4�/�#�����#�/�5�<�<�<�<�<�<������������ ����������¿¸·¿��������������#�+�(�'�!���
������������������!�-�:�G�R�V�S�F�:�-�!���������y�����������������y�u�o�l�i�f�^�`�b�l�y�y�x�p�w�y�������������y�y�y�y�y�y�y�y�y�ּ��ܼؼܼ����ּʼ����������μӼּ'�4�Y�f�r���������r�f�Y�@�4���� �'E�E�E�E�E�E�E�E�E�E�E�EuEiEfEcEZE^EhEuE��@�M�Y�`�f�l�f�Y�M�C�@�4�*�'�&�'�4�<�@�@��!��������������������L�r��������������~�r�e�Y�@�'����7�L 7 9 & @ , D R d , U i Q N a \ 9 F X O I I  @ H 5 K | C j > . , Y B < " U t   o F 9 1 : # # K  / C ] k J v V @ [ n c      �  <  �  �  D  �  �  �  W  %  Q  x  U  �  :  g  9  y     "    �  e  �  �  �  {     a    �  �  �  �  
  �  U  V  O  �  r  /  �  �  �  �  �  �  �  �  �  a    �    �  ^  �t��49X<t�;�`B<�o�D��<�<�o<�/;o>o<t�<u<D��=��w=D��<��
=��<�j=\)=H�9=,1<�`B=t�=L��=�7L<�h=�P=T��=<j='�=\=t�=�o>1&�=<j=���=49X='�=#�
=Ƨ�=�1=u=<j=�o=P�`=aG�=@�=�\)=�hs=�1=���=��=��-=��=��=��=�->,1BaB�oB
C�B��Br�B��B�B ̟B��BF�B	$DB�6B�BoyB_�Bw,B�SB0�BQ�B��B�B=�B!��B.�A���B%ϵB!B�B A8B"IwB��B
��B��B^nB�NB4B�`B��B��B
�B[�A���B�
BY�B�BB��BW�B��B�8B)�B�B,��B/�"BJ>B��B�B�B�
B��BD5B��B
@B*�B�B��BFB ��By�BC`B	9�B�&B��BUB�B@�B�B;�B~(B��B?�B	�B!u�B
�	A��B%�JB!
�B @"B"@B��B
��B�5B@�B�rB?�B�GB�B�,B�aB�hA��SB�B@<BĸB��B��BG�B�B�BFvB��B,C�B/�{B>�B��B�;B�.B<�BĴBLPA��NA�C3A��CC�q�A�āA��P?��A��A��`A�>�At�A�zBtaA��}Ac�A��qA���A���A���A��LAj%A_t�AH�6A��P@�@���@c1NA��@~�A��A���A_NA��C���A��?B�Ax&�A�g)A�:A�]�A�
�A�$AIVA�5�A�/�AA�9A��A� �A���@ko�A�EA�|@�[�@��8C��i@�X+@��S?�t~B�A���AɀA��AC�\wA�mUA���?�%�A��5A�;A�x{AslIA�ZB�$A��AeA�pBA�X9A�{?A�A�X�AjA^ԅAG�HA��"@��J@�8A@d�A�~@1A���A�T.A`C�A�e=C��*A��B)�Ax�AA�t�A̋�A�
A��A�t=AH�=A�TA�]�AB�LA���A�>yA���@lJYA�A"�@� H@όC��O@��@�c?��   
                  #            �      	      E   %   	                  	         .                  C      !   �      G            =   /                                       <   !         .               !      )      +      +            3   %      %                        /                  )      !   !      /               '         #                                          %            !   !      %      )                  '                                 )                                                '         #                                          N�^UN��N��UO���O�O�N2��O�<N@��P�[N(ǟOW��N(�3NA�NȹO�SO�"N1��OP��NEIN�$<O��lOh�)O5,�O��-O'�O�e�N+�[Nf��N�9.O%b	N���O���N��"O�˩OuB/N���O���N��N'xNa�O)EP�)N�ޮN�xVOɓ�N�h�O#�N3�.OHd�N�I�OF{�N��N2"�N�&O�~�Om}�N�n�N�Oxif  �  I  3  `  ~  u  �  �  %  �  W  S  J    �    5  >  �  :  '  ^  2  �  A    �  �  �  �    )  :  n  e  �  �  �  �  �  	�  W    }  t  {  �    �  �  %  O  �    
_  M  �  �  ���C���C��e`B�D���D���ě���o;ě�;D��%   =]/;ě�;�`B<#�
<��<�t�<D��<�j<�o<��
<���<�9X<��
<�1<�/<�h<�j<���<�`B<�/<�h=P�`<��=C�=�+=C�=�%=\)=\)=t�=y�#=��=�w=�w=#�
=#�
=,1=,1=<j=y�#=]/=�o=y�#=�o=���=���=��
=��
>+����������������������������������������mjjnty�����������tmm;=IUZahnz�����znaUH;QPRanu�����������znQ  sn|����������������s������������������������������������������������������������IFGINN[gt|�����tg[NI$)6BFIB;61+)$$$$$$$$��������������������������&(#���������59HGEB4)	���������������������)5BNORRPHB>5)#KOS[hooh[OKKKKKKKKKK���������������������������������������������
#/2464/#
 ����������������������gkt��������������wkg "/3;?HHIHGA;/" #06:B[caYI0#������������������������������������������������������������MKSZ[bht}��|wtrh[TOM{y|���������������{{�������

������"#
5BN[kt{���|tg[NKC<65��������

�����������������������		)5=CFGFB5)	��������������������85:<HLRNH<8888888888;<HU[\UHG<;;;;;;;;;;9767;HLTY^bcbaVTHF;9����������������������������������������g`\\bhltx{}tqhgggggg���)5?C?5-)���� �
%&#      #/<BHKIH@<;/-'#������������������)56;>@>:5) ������������������������������������������������������66**.1666666666��������������������MLKKP[hptwzy~~{th[PM�������

����������������������������������������������������$*(!	����=�I�T�V�b�l�b�]�V�I�=�5�0�.�0�;�=�=�=�=���������������������������������n�zÇÓÞàáàÜÓÈÇ�z�z�p�n�k�h�n�n����������������������������������������E�E�E�FF	FFFFE�E�E�E�E�E�E�E�E�E�E��������������s�p�s�u�������������������������������������������������������������3�=�@�A�@�8�3�'���'�-�3�3�3�3�3�3�3�3�g�����������������s�g�Z�A�/�'�(�1�>�N�g�����������������������������������������6�B�O�[�h�h�t�r�h�[�R�O�B�6�3�-�+�+�/�6����������������������������������������Ó×ÖßÓÇ��z�u�zÇÌÓÓÓÓÓÓÓÓ������������������������������������������"�;�T�a�p�r�l�a�T�H�;�7�"�	����������G�T�d�m�o�u�n�m�`�G�;�4�"���	���.�G�O�U�[�S�O�B�6�4�6�9�B�H�O�O�O�O�O�O�O�O������+�5�:�8�5�(��������������H�K�U�]�a�U�H�?�>�D�H�H�H�H�H�H�H�H�H�H�/�<�H�K�I�H�?�<�3�/�+�#�!� �#�$�/�/�/�/��(�5�A�N�[�f�f�e�Z�N�A�(���
�����`�m�y�����������~�y�m�`�T�F�?�>�H�T�Y�`��"�'�.�;�>�=�A�A�;�9�.�"��	����	�����������������������v�s�n�o�s�y������	��"�/�;�A�G�H�K�H�>�;�/�'�"��	���	������ʼּ����ּʼ�������n�^�`�q��ûλлѻлʻûûû����������ûûûûûû���!�+�,�!��
��������������ìù��������ùíìàÓÇÇÇÈÓàçìì�-�:�F�S�V�_�e�e�_�S�F�:�-�$�!���!�)�-�"�/�;�H�H�M�H�C�;�3�/�"�������"�"���	��"�&�*�(�%�"��	�������������������.�1�;�?�;�2�.�"��	���"�,�.�.�.�.�.�.�|�t�n�g�[�N�B�5�+�$�#�'�)�5�N�gD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D���*�6�?�C�K�O�C�6�.�*�(��������ƎƚƧƳ����������������Ƴƚƍ��z�yƀƎ�������Ŀʿѿݿ��ݿѿʿĿ�����������������������������������������������������ùûû��ùìãéìøùùùùùùùùùù�������
��&�#���
�������������������������)�6�@�O�B�7�3��������øóú�����h�tāčĔĚěěĚčā�t�h�g�[�W�[�f�h�h�������������������������~�������<�I�X�b�i�j�b�U�I�0��
���
����#�0�<���������������������������������������ؾf�s�����������s�f�Z�X�M�A�A�G�M�Z�`�f�<�=�<�7�4�/�#�����#�/�5�<�<�<�<�<�<������������ ����������¿¸·¿�����������
������
��������������������������!�-�:�G�R�V�S�F�:�-�!���������y�������������}�y�x�o�l�k�d�l�q�y�y�y�y�y�x�p�w�y�������������y�y�y�y�y�y�y�y�y�ʼռּټ߼���ּʼ����������������żʼ4�M�Y�f�r�����{�r�f�Y�@�4�'����'�4E�E�E�E�E�E�E�E�E�E�EuEiEgEdE[E_EjEuE�E��@�M�Y�`�f�l�f�Y�M�C�@�4�*�'�&�'�4�<�@�@��!��������������������L�e�r�~�������������~�r�l�e�Z�L�=�2�@�L ( 9 ! A , D S < * U # Q N a \ 5 F B O L I " @ H * F | C h > .  A 8 8 " 0 o  o 8 9 / : # # E  / * ] [ J Z N @ [ n W    �  �    ;  �  D  v  Z  �  W  �  Q  x  U  �  �  g  �  y  �  "  �  �  e  i    �  {  �  a    t  �  �  �  
  T  	  ;  O  {  r  "  �  �  �  s  �  �  �  �    a  A  #  �  �  ^  
  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  �  �  �  �  �  �  �  �  �  �  l  S  9      �  �  �  �  t  I  D  @  ;  2  (      �  �  �  �  �  �  �  �    h  Q  :     .  .      �  �  �  }  V  *  �  �  f  �  j  �  )  x   �  R  Z  _  `  ]  V  K  ?  2  '    �  �  �  �  j  B    �  D  ~  m  Z  C  "    &  9  $  	  �  �  �  y  X  ^  6  �  �  �  u  s  r  q  p  n  m  k  h  e  b  _  \  Y  T  P  K  F  B  =  �  �  �  �  �  �  �  X  9  $        �  �  `    �  9  |          +  [  e  s  �  �  �  �  �  �  �  r  L  �  �  L  �      $      �  �  �  �  d  7    �  �  8  �  �  �  e  �  �  }  v  p  i  b  [  U  N  F  <  3  )            �   �  3  c    �    �  '  Q  V  A    �  �    5  J  "  	�  �    S  M  H  B  <  6  1  '         �   �   �   �   �   �   �   �   �  J  F  B  A  A  K  m  �  �  �  �  �  �  �  �  �  �  �  �  �           �  �  �  �  �  �  �  �  �  �  �  �  y  l  _  R  �    n  �  �  �  �  �  �  �  �  �  b    �  e  �  �  ,  �  �  �  
             �  �  �  �  f  3  �  �  A  �  \  }  5  S  q  |  |  {  u  p  i  b  [  S  H  $  �  �  z  -  �    �  �    ,  9  :  :  >  <  /    �  �  �  R    �  Q  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    0  9  :  7  ,    �  �  �  l  6  �  �  �  D    �  /  '      �  �  �  �  �  �  j  <    �  �  6  �  D  �  M  F  $  Q  ^  Y  R  F  7  !    �  �  �  �  q  R  1    �  �  �  2  2  1  -  '  !        �  �  �  �  �  �  �  p  P  #   �  �  y  n  a  R  C  2     
  �  �  �  �  \  -  �  �  �  ]  7  �    3  ?  >  -    �  �  �  q  ;     �  n  �  V  �  �   �  �  �        �  �  �  �  �  �  �  �  �  M  �  �  �  6  O  �  �  �  �  �  x  h  X  H  8  '      �  �  �  �  �  �  p  �  �  �  �  �  �  �  �  |  Y  0    �  �  =  �  �  $  �  X  �  �  �  �  �  �  �  X  &    �  �  A  �  U  �  �       4  �  �  �  �  �  �  w  i  W  E  1    �  �  �  k  %  	  �  �      
    �  �  �  �  �  �  �  q  I    �  �  �  N    �  �  �  �      "  )  %      �  �  �  �  e    �  �  �  "  !  )  1  8  1  '        �  �  �  �  �  �  _  8     �   �  Q  l  j  ^  N  =  ,    �  �  �  �  S  ,  �  �  ]     {  w  �  /  �  B  _  a  J    �  v    �  �  6  C  �  X  	f    K  �  �  �  �  �  �  ~  t  k  `  P  ;    �  �  �  L    �  K  �  �    L  l  z  �  �  �  �  x  ^  %  �  l  �  b  �  r  �  �  �  �  �  �  l  R  2    �  �  �  �  �  �  ^  )  �  �  s  �  �  �  �  �  �  �  �  �  �  �  �  �  f  B    �      0  �  �  s  [  C  .        �  �  �  �  �  |  `  B  %     �  �  	  	[  	�  	�  	�  	�  	�  	�  	�  	�  	�  	%  �    m  �  �  A  �  W  .    �  �  9  �  �    �  �  C        �  �  �  �  �        �  �  �  �  �  �  d  <    �  �  :  �  �  0  �  �  }  w  q  j  ]  P  C  4  &      �  �  �  �  }  J     �   �  t  t  q  e  X  J  ;  *    �  �  �  �  _  0    �  �  v    {  u  n  d  V  D  ,    �  �  �  �  �  �  r  =  �  �  M  �  �  �  �  �  �  v  b  N  8  "  
  �  �  �  �  �  k  -  �  a    
    �  �  �  �  �  �  �  z  j  q  �  �  �  �  �  �  �  �  �  �  y  j  [  P  D  6  $    �  �  �  �  N    �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^    �  �  (  %         �  �  �  t  E    �  �  x  @    �  f  �  c  �  �    $  5  D  N  O  L  E  :  )    �  �  �  l  Z  7  �  �  �  �  �  �  �  �  �  �  �  �  �  {  q  g  ^  S  I  >  4  )  �  �  �  �    �  �  �  �  ~  o  Q  '  �  �  d    �  �  N  	�  
   
H  
\  
^  
N  
-  	�  	�  	m  	  �    u  �    #  $  �  �  M  L  F  8  *    	  �  �  �  {  H    �  �    �    �  �  �  �  �  �  �  �  �  �  �  e  H  )    �  �  t  B    �  �  �  �  �  �  �  �  z  m  `  P  @  1  %    
  �  �  �  d    J  c  q    �  ^  '  �  �  O  	  �  �  [    �  �  )    �