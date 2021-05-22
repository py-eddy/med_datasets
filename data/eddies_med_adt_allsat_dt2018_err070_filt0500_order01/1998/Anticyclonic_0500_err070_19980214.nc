CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��vȴ9X      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mۗ)   max       P��s      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��hs   max       >I�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @E�Q��     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vm\(�     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @N@           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�y           �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       >L��      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�k   max       B,/�      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B,~a      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�=   max       C�XH      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�   max       C�Ym      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          |      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mۗ)   max       P.�s      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?��D��*      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��hs   max       >I�      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @EǮz�H     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vm\(�     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @N@           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�/�          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��-V   max       ?��D��*     `  U�         O               _         =   |         R         
   )   	      '         B   o   0         %      )   )               (      	         G                              )            "      ,   L      ONாP��sN$�_Nz��N��O���P��O�e+N2@�O���P}�M�O�OBP@h[N:y@N��OR�O�D�N!
dN�4�O��eNF�N^Y�P=��P�:O���N��N^˰O#0�OL4�O�/5O�*�N�a�N�tTN�C�N��dO{e�M�&tNYF�NH�lN�5P��NNFOe��N���N�fMۗ)N��qO��gN*x�N|�O^f�NX�OoFM��O�R�N�4�O+�|Oˡ�N}�TN/�`��hs��/�D����`B�ě�;o;�o;ě�;�`B<t�<#�
<49X<49X<49X<D��<T��<e`B<�o<�o<�o<�o<�C�<�C�<���<��
<�9X<�9X<�j<�/<�/<�/<��=o=C�=t�=t�=�P=��=�w='�=,1=0 �=T��=T��=e`B=e`B=e`B=u=u=y�#=�%=�o=�+=�C�=�hs=�t�=��P=�-=�"�>   >I�>I���������������������BNR[[\gt}�~tg[QNBBBB�����/=HZz|o]H/���()46=BOOOFB=6)((((((��������������������#0400##0<IUbcdb[K<0#���������������������������������������������������������������������
���������&5B[gt{{yqjB)�����������������������������������������������
#+AUeia<����:<HJUXadaUQH?<::::::9<=ABNOSTTNB99999999����������������������������������������������������������������������������������
#/<UTZ\UO/#
��rtz~�������trrrrrrrr��������������������226AUz�������z_UH>>2�������AHIA5����������������������������������������������������������������������! !��  &)46BHOQX[^[VO6))%$(<QTUadelrvkaH</)
	).8;;=94
������������������������������������������������������������eb`ehst�������theeee@>DHKTamz�����zmaWH@?BNNNV[`[NIB????????25<BNOXWNB@522222222��������������������������������������������#-7BWN@5)����\`gtxxttsg\\\\\\\\\\�����������������
#$&%#
������������������������
 �����
xuuy{����������{xxxx��������*���������������������������#$0020.#"/4;HRTUTOHG;/#��������������������[TZ\_abmz|���|zoma[[��������������������56@Ngt�������ti[NB95��	�������������������������vqnnw��������������v������������������������������������������������������������������������������������������������������������������������	�/�H�[�m�v�q�a�H�/�������������������	�;�=�H�J�H�C�?�;�9�/�-�/�0�5�;�;�;�;�;�;�(�4�A�D�M�R�M�I�A�5�4�(�"�%�(�(�(�(�(�(�'�1�4�?�=�4�'�'��$�'�'�'�'�'�'�'�'�'�'�����������������������f�W�K�O�Y�f�r��Z�g�s�����������������g�A�*���"�(�A�Z����(�+�,�(�'�������ݻڻٻܻ޻��׾�����������׾ӾҾ׾׾׾׾׾׾׾�������������������������������������)�F�[�t�m�t�o�c�O�6��������������¦©²º²¦���������ûǻ»�������������x�m�p�x�������������
�������þ������������x�ÇÊÓÓÕÓËÇÅ�}�z�x�z�ÇÇÇÇÇÇ�{ŇŔŠťŦŠŔŇ�{�q�o�{�{�{�{�{�{�{�{�s���������������~�s�j�f�`�Z�U�Z�f�p�s��������������������ŹŭŠŕřŠŮ�������<�@�B�>�<�/�*�%�/�3�<�<�<�<�<�<�<�<�<�<���������������������������������������ҿT�`�m�~���������m�`�T�G�>�;�3�4�7�=�G�T�ܹ�����������ܹ۹йֹܹܹܹܹܹܹܹ�¿��������������¿´²­°²·º¿¿¿¿�5�A�N�Z�j�y�y�s�n�g�N�(����������5ĳ�����
�#�9�8�����ĳĚĎĂāĎĞğĦĳ�Ŀ����(�:�5�(��������ѿĿ�������������������������������������������������*�6�:�9�:�6�-�*��������������ûлܻ����������ܻлû�����������'�4�M�V�f�h�m�f�V�M�@�:�4�-�'�$�����������)�6�B�>�6�)����������������ƧƳ�����������������������ƳƧƔƚƧ��� �!�*�,�&�!������������������)�*�5�)���������������Z�_�f�k�f�e�f�m�f�Z�M�J�A�5�;�A�M�V�Z�Z�F�S�_�l�o�l�k�a�_�S�I�F�@�A�=�@�F�F�F�FāčĚğĦĭĳĶĴīĦĚčā�z�p�q�t�|āƁƈƎƎƎƁ�y�u�q�s�u�yƁƁƁƁƁƁƁƁ�v�~ÇÈÓÙàäàÓÇ�~�}ÂÇÇÇÇÇÇÇÇE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������ÿſ¿������w�m�h�T�K�F�K�W�m�y���m�y�|�~�y�m�`�`�`�f�m�m�m�m�m�m�m�m�m�m����������������������������������������ǡǡǭǭǮǭǬǣǡǔǈǁǁǈǌǔǙǡǡǡ�ʼּ��������ּ̼ʼƼɼʼʼʼʼʼ�āā�t�p�h�d�h�tāāĂāāāāāāāāā�нݽ�������������ݽнϽȽнннн����������������������������y�o�o�t�����`�T�G�?�=�G�T�`�b�`�`�`�`�`�`�`�`�`�`�`�������������������������������������������
��#�)�.�0�2�0�#���
���������������z�����������������z�y�w�z�z�z�z�z�z�z�zŔŠŭŹ��������������ŹŭŠŝŔŒőŔŔ��!�"�-�:�?�:�-�!������������	��"�+�/�,� ��	���������������������	�`�h�c�_�Y�S�G�F�:�5�-�,�-�6�:�F�N�S�_�`���ʼּ����������ּʼ������������x�����������������������x�l�_�T�P�P�`�x�U�a�f�j�a�U�H�<�2�<�H�J�U�U�U�U�U�U�U�U���#�*�#���
���
��������� 5 R B b 4 B 1 C " N 1 4 S ? O Y W 0 : r K 9 H n O < L S : V ; 9 Q : 9 . I 3 V P t B 6 B 9 a " t e y & 6  y B w 6 h ) $ S =  9  (  1  h  �  G  �  �  #  b  �    -  _  �  y  �  =  ]  �  �  �  k  �  �  �  �    |  �  �  u  s  �  �  �  �  �  ;  �  �  "  �  @  �  �  �  M  �  �  B  �  �  h  ^  D  �  �  s  �  �  @����ě�=�+�D��:�o;ě�=o=���=��<u=�hs>	7L<T��=C�=�v�<���<���<���=ix�<ě�<�`B=e`B<�j<���=�{>1'=�O�=t�=+=�o=T��=�hs=��=@�=@�=L��=P�`=��-=0 �=H�9=aG�=L��=��#=aG�=��T=�\)=�C�=�7L=��-=�^5=��=�t�=�"�=��P=� �=��-=�"�=��>��>L��>t�>bNB҉B	,�BMFB�]B"-�B%BEB&<^B�B ՘BB|�B^�BqB"�B�0Bq]B��B�uB�B�5BO�B	�B�jBqSB�B6�BǧB�tB$iB�rB��B�BIB nBwvBJB[�A�'oB=BٚB!�UB�OB�B	e�BM�B��B!<�B�,B)D�B,/�B!�B%gA�kB5MA�hpB*�KB	�B�B*�B�B��BK>B��B	?�BAzB��B"9�B%D�B%��B��B!=�B?�BAcB�UB='B"�BB�]BA�B��B��B�MB��B��B;�B�UB�BC�B>�B�vB��B�~B;�B�WB/B?�B�NBRB��B�FA�p!B5JBAB"@B��BLB	�`B��B?�B!='BK1B)BGB,~aB!�"B%=�A�~�BCA���B*�"B	>pB9�BA�B�@B�-B?�A�f�A��A�}A�o�A9��@�^@�#A��@�MAAU��A�R�Aպ�A�}a@�8�AOԂA�t�A��!AClBA��6A�(�A�֣AhO�?�=A��<A���A��A��)A�~A�~�@���@Ч�A��B��@^�[A��A>!@�>�A�!BhA�?�A�.C�XHAo7,AkZ�A��Bb'A	�Aܘ�A-%A �Afr!AI��A���A���A��@peA���@�(�A ��@���A�)�A�\�A�qoA��WA���A���A9Xf@��@��FA��p@�_AV��A�N�A���A���@��%AO�A�~�A��VAC2A���AA�Z�Ah��?�A�]A��IA�_A��A���A�y@� @�MAҁ�B�T@\J}A�z�A=��@���A�tLB>{A�%A�s{C�YmAq ~Ak�QA�j�BAA	%A�}�A.�"A�AfOHAI/wA苠A�}�A��u@stA��"@���A �@��mA�i&A�n�         O               _          >   |         R            *   	      (         B   p   0         &      )   *               (      	         G                           	   *            "      ,   L               ?            #   5            3         1                              /   7   #                                                +                     %                              !               /               !            +                                       !                                                      !                                                   !      ON�q�P.�sN$�_Nz��N��O\@�O�`�OW�N2@�OnOP-��M�O�N���O�ON:y@N��N�� O:�N!
dN�4�O��NF�N6�zO�9�O�XxO"rN��N#��N�OL4�NɗJO��N�sN�tTN�C�N��dO?ִM�&tNYF�N-n>N�5O�<�NNFOe��N���N�fMۗ)N��qOZ0!N*x�N|�O^f�NX�OoFM��O�R�N�4�OI�Oˡ�N}�TN/�`  �  P  >  �  r  !  8  �  �  �  	  
�  �  �  	5  5  �  |  P  �  �  '  H      	�  o  �  �  	y  m  �  F    &  )  �  x  p  *  �  �  �    q  E  �  �  �  �  �  �  �  �  �    �  E  	4  �  3  %��hs��1<�C���`B�ě�;o<D��=49X<e`B<t�<���=t�<49X<u=<j<T��<e`B<�C�<�/<�o<�o<���<�C�<��
=��=���=�w<�j<�`B=+<�/=H�9=�P=\)=t�=t�=�P=49X=�w='�=0 �=0 �=�C�=T��=e`B=e`B=e`B=u=u=�\)=�%=�o=�+=�C�=�hs=�t�=��P=�-=�G�>   >I�>I���������������������[NV[gtvztrg[[[[[[[[[���
#/;GUcebH</	�()46=BOOOFB=6)((((((��������������������#0400# #0<IU[]ZTLB<0+%$ ������������������������������������������������������������������������������)5N[gmssgcWB)���������������������������������������������
#/7=AB=/#
 �:<HJUXadaUQH?<::::::9<=ABNOSTTNB99999999�������������������������������������������������������������������������������� 
#/<@IPRSH</#rtz~�������trrrrrrrr��������������������C>=BHbz��������naUHC�������%(!�������������������������������������������������������������������	���  &)46BHOQX[^[VO6)4007<HOU]]USH<444444
)+58882)
������������������������������������������������������������eb`ehst�������theeeeMIJPTaimrz����}zmaTM?BNNNV[`[NIB????????25<BNOXWNB@522222222���������������������������������������������%,264*���\`gtxxttsg\\\\\\\\\\�����������������
#$&%#
������������������������
 �����
xuuy{����������{xxxx��������������������������������������#$0020.#"/4;HRTUTOHG;/#��������������������[TZ\_abmz|���|zoma[[��������������������56@Ngt�������ti[NB95��	�������������������������vqnnw��������������v�������������������������������������������������������������������������������������������������������������������������"�;�M�Y�Y�Q�;�/���������������������;�=�H�J�H�C�?�;�9�/�-�/�0�5�;�;�;�;�;�;�(�4�A�D�M�R�M�I�A�5�4�(�"�%�(�(�(�(�(�(�'�1�4�?�=�4�'�'��$�'�'�'�'�'�'�'�'�'�'�����������������������r�f�`�V�Y�f�r��s�������������������s�g�Z�R�J�K�N�V�g�s������!�&�(�'�!���������߻����׾�����������׾ӾҾ׾׾׾׾׾׾׾���������������������������������������9�P�W�\�b�`�[�O�6�������������¦©²º²¦�����������������������������x�v�x�{���������ʾ׾������׾ʾ�����������������ÇÊÓÓÕÓËÇÅ�}�z�x�z�ÇÇÇÇÇÇ�{ŇŔŠťŦŠŔŇ�{�q�o�{�{�{�{�{�{�{�{�������������y�s�n�f�a�Z�W�Z�\�f�s�x�������������������������������ŹůŬŽ���<�@�B�>�<�/�*�%�/�3�<�<�<�<�<�<�<�<�<�<���������������������������������������ҿT�`�m�v�|����}�y�m�`�T�G�@�;�9�;�@�G�T�ܹ�����������ܹ۹йֹܹܹܹܹܹܹܹ�¿����������¿³²±±²¸½¿¿¿¿¿¿�(�5�A�N�T�g�o�n�i�Z�N�9�(�!�����!�(�����������	������������ĿĺķĺĿ���̿ݿ���������������߿ݿ׿տݿ�������������������������������������������*�6�7�6�6�6�6�6�6�*����������ûлܻ���� �����ܻڻлû����������ü�'�4�M�V�f�h�m�f�V�M�@�:�4�-�'�$�������������
�����������������������Ƴ�������������	������������ƳƨƚƤƳ����!�$�*�!�!�!�����������������)�*�5�)���������������Z�_�f�k�f�e�f�m�f�Z�M�J�A�5�;�A�M�V�Z�Z�F�S�_�l�o�l�k�a�_�S�I�F�@�A�=�@�F�F�F�FāčĚĦĩİĳĴĳİĦĚĐčā�}�t�u�{āƁƈƎƎƎƁ�y�u�q�s�u�yƁƁƁƁƁƁƁƁ�v�~ÇÇÓ×àãàÓÇ��ÅÇÇÇÇÇÇÇÇE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��y�������������������y�m�`�T�R�V�[�b�m�y�m�y�|�~�y�m�`�`�`�f�m�m�m�m�m�m�m�m�m�m����������������������������������������ǡǡǭǭǮǭǬǣǡǔǈǁǁǈǌǔǙǡǡǡ�ʼּ��������ּ̼ʼƼɼʼʼʼʼʼ�āā�t�p�h�d�h�tāāĂāāāāāāāāā�нݽ�������������ݽнϽȽнннн����������������������~�y�s�u�{��������`�T�G�?�=�G�T�`�b�`�`�`�`�`�`�`�`�`�`�`�������������������������������������������
��#�)�.�0�2�0�#���
���������������z�����������������z�y�w�z�z�z�z�z�z�z�zŔŠŭŹ��������������ŹŭŠŝŔŒőŔŔ��!�"�-�:�?�:�-�!������������	��"�+�/�,� ��	���������������������	�`�h�c�_�Y�S�G�F�:�5�-�,�-�6�:�F�N�S�_�`���ʼּ�������߼ּмʼ������������x�����������������������x�l�_�T�P�P�`�x�U�a�f�j�a�U�H�<�2�<�H�J�U�U�U�U�U�U�U�U���#�*�#���
���
��������� 5 9 P b 4 B - .   N   D S ,  Y W ( & r K 4 H w J 7 & S F E ;   Q ? 9 . I % V P ` B , B 9 a " t e k & 6  y B w 6 h # $ S =  9  �  .  h  �  G  �  �  �  b  �  #  -  �  %  y  �    �  �  �    k  �  G  	  \    Y    �  �    �  �  �  �  �  ;  �  r  "  �  @  �  �  �  M  �  5  B  �  �  h  ^  D  �  �  M  �  �  @  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  �  �  �  �  �  �  �  {  v  p  h  `  Y  Q  H  @  7  .  %  -  0  5  ;  B  L  L  @  '    �  �  {  D  �  �  7  �  Q  �  �  +  �  �  �  #  7  =  3    �  �  n  "  �  p  �  g  �   �  �  �  �  �  �    q  d  V  I  :  (       �   �   �   �   �   �  r  j  b  Z  Q  G  :  )      �  �  �  �  �  �  }  u  q  l  !            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      .  8  4  '    �  �  s  3  �  �  �  �  �  V  '  z     �  �  <  i  �  �  �  �  s  T  ,  �  �  )  �  �  )  P  h  �  �  �  �  �  �  �  �  �  s  L    �  l  �  h  �  �  �  �  �  �  �  �  �  �  u  e  U  B  ,       �  �  �  �  �  l  �  p  �  �  	  	  	  �  �  �  �  A  �  �    h  �  �  b  *  	�  
W  
�  
�  
�  
�  
�  
t  
  	�  
b  
z  
h  
%  	�  	u  	  G  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  F    �  �  �  �  �  5    �  �    s  �  �  	$  	4  	  �  }    �  =  �    /  %  �  5  /  (  "          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  j  Z  J  ;  *    
  �  �  �  �  �  �  �  u  w  z  |  {  y  r  j  `  U  I  9  &    �  �  �  �  l  E    �  �    3  I  O  J  8    �  �  �  4  �  k  �  �    �  �  �  �  �         &  5  /  &      �  �  �  �  �  �  �  {  �  �  �  �  �  z  `  F  -    �  �  �  {  Q  +    �  �  �  �  �    $  '  $      �  �  �  �  U  
  �  b  �  �  �  �  H  A  9  1  '        �  �  �  �  z  N  !  �  �  �  [  %            �  �  �  �  �  �  �  v  `  H  1      �  �  B  u  �  �  �      �  �  �  �  e  7  �  �    {  �  �    �  �  �  �  �  �  �  	  	L  	�  	�  	�  	�  	�  	p  	    �  v  	  �  *  B  T  U  O  ]  m  a  =    �  �  O  �  Y  �  �  �  *  �  �  y  e  R  >  *      �  �  �  w  8     �  �  O     �  �  �  �  �  �  �  �  �  �  �  �  z  ^  ;    �  �  o  0   �  �  	  	n  	y  	r  	[  	5  	  �  �  H  �  �  \    �  Y  �  �  9  m  I    �  �  �  d  1  �  �  �  t  L  $  �  �  _  �  =  �  �  �    5  W  n  }  �  �  �  �  z  \  *  �  �  *  �  �  �    8  E  A  8  +        �  �  �  a    �  �      �   �          �  �  �  �  �  �    p  _  M  9  �  g    �  {  &  	  �  �  �  �    ]  ;    �  �  �  �  s  Z  F  7  t  �  )          	  �  �  �  �  �  d  >    �  �  t  9  �  �  �    u  l  d  J  /    �  �  �  v  E    �  {  :  �    R  �  :  g  x  q  b  L  )     �  �  A  �  w  �  V  �  �  	  ;  p  j  d  ^  X  T  T  T  T  S  Q  K  E  ?  :  3  ,  &      *        �  �  �  �  �  �  �  �  ~  n  ^  P  B  9  7  5  �  �  �  �  �  �  �  �  a  <    �  �  �  m  <  �  �    �  �  �  �  �  �  �  �  �  �  y  p  g  \  P  D  7  *    	  �  (  �  �  �  �  �  �  �  |  3  �  �  M    �  0  �  �  �  �                �  �  �  �  �  �  �  �  �  �  �  �  �  q  E  2    �  �  �  @  �  �  X    �  Z    �  �    �  �  E  #    �  �  �  �  �  �  y  ^  @  -  �  �  �  <  �  �  Z  �  �  |  i  Q  8       �  �  �  n  M  0    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  R  :  %            �  �  �  �  �  �  �  �  �  e  P  -  �  �  �  6  �  �  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  M    �  �  �  R  �  <  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  x  q  �    d  K  4      �  �  �  �  �  {  _  C  "  �  �  �  y  �  m  f  ]  R  C  *    �  �  �  B  �  �  .  �  >  �  8  �  �  �  �  �  �  �  �  �  �  �  �  E  	  �  �  g  2  �  �  �  �  �  f  F    �  �  �  i  L  i  R  2    �  �  �  v  �  H    �  �  �  �  �  �  �  �  i  M  2    �  �  �    `  @  !  �  �  �  �  m  =    �  �  ^    �  m    �  Y    �  �    E  &  �  �  �  r  >      �  �  3  �  �  F  �    �  �  `  	  	/  	*  	  �  �  �  ~  S  '  �  �  �  W    �  �      �  �  �  �  X  +  �  �  x  %  
�  
=  	�  	  |  �  �    �  �  �  3  #      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  %    
  �  �  �  �  �  v  S  ,  �  �  �  z  G    �  �  r