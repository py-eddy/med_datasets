CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�^5?|�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�@�   max       P�7E       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       =��       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @EXQ��     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @vu\(�     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q            �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�4`           7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��9X   max       >�w       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B*S       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��    max       B*;B       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C�v       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?($i   max       C�h�       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          h       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�@�   max       P[.h       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�Q   max       ?ܯO�M       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       =�-       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @EW
=p��     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vt�\)     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q            �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0�   max       ?ܭ��U�=     �  \8                                             #            &         >                     D      *                        	         -                     %   L                   #   )   '   0      )            %   g   N      NX��O`��N@�nNjЪOO_N��kN�yN8�.M��xNY�N.WMONՈO9�P>4�O1*�N�_�NKdO�#RN���O��'P�7EO��LP��O!N�tbNu�ON�PD�dN�
PG��OuEmN�M�@�O^InO�01N���NW�"Np�OJ�O>?�O��Oum�O33�N`R�Nk�O��=OOĉEO�#sNA6jN�8N�0O�o�O 8�Oկ�O�0GO���O��GN�KO���O
dN^�DN%�)OM�!O�"O��5N<�~N��
��`B���
�#�
�o�o�D���o��o��o��o%   %   :�o;ě�<o<o<o<49X<T��<T��<T��<e`B<e`B<e`B<u<u<u<�o<�o<�o<�C�<�t�<�9X<�9X<�j<ě�<ě�<ě�<���<���<���<���<�/<�/<�/<�/<�`B<�`B<�h<�<�=o=C�=C�=C�=t�=�P=��=�w='�=49X=49X=@�=@�=D��=Y�=�C�=�t�=��	 ���������	
����������������������������������������������?<DN[_gotxztog[TNEB?fgit��������tlhgffff����������������������������������������

��������������������~{��������~~~~~~~~~~��������������������������������������a^ho}����������|ytha����������������".5BNPTUTNLB5)")16<BEHB<62) ���������������������	"/;;@CMC;/"�� �
 
      #&/:HTab_UH<3/*�����#Hanz��RH<
���ZXbnz���������tnolaZ]b`faez���������zoa]35>BN[glmg[WONB>6753������������������������������JDEDFNT[gkrpig[NJJJJ��������� ����������olqt���������toooooo�����*<Umpm[UD0#���AACLT_efhkm|�~xth[OAqotsotw����{ttqqqqqq##0110%###########157>@?85)z�����������������}z������������������������������������������������������������\\]agkt������ytg\\\\������������������������������������������������������������������  ������
!
���������*$/<DHKHG<0/********����5NZSK5)����pptu������������wtpp1./4<CKV[`djeddb[B51	)6CJMKD866)�����



����������������������������GEEGMNY[]ded`[ZNGGGG#/3<HU[`ceisnaH/#,)'')/0<@GIJJIIEA<0,�����)6===1) ��  )7BO[d[OL>6)  )5BNUWTHB) �����)=HLHB)��*%&(/<HKTSLH>><;:/**�����").25:5)�>:9:<BNPV[^eeb\[NJB>jkiimz~|zzmjjjjjjjj++-/<==></++++++++++�������������������

�������������������������56@BCO[XOB=655555555��������������������àìù��������ùìàßÖàààààààà�������ûͻٻػлû������x�i�a�l�x�������������������������������������������������������������������������������������)�5�B�E�B�:�5�2�)�����������n�t�zÅÁ�z�n�g�a�`�U�N�N�U�a�f�n�n�n�n�����
���
����������������������������m�y�����������~�y�p�m�g�m�m�m�m�m�m�m�m¦¬²´²¦¦¦¦¦¦¦¦¦�����������������������������������������<�I�M�K�I�<�0�%�0�7�<�<�<�<�<�<�<�<�<�<��"�#�)�-�-�/�1�3�/�"���	�������	���#�/�<�H�S�R�H�B�<�/�#�� �"�#�#�#�#�#�#�ܻ������������ܻлû��ûлԻڻ���*�O�hƉƚơƎƁ�u�h�C��������������h�u�yƁƎƓƜƝƚƎƁ�u�j�h�d�\�Y�\�a�hÓÙÙàæàÙÓÇ�~�z�w�zÁÇÎÓÓÓÓ����������������z�w��������������������a�m�v������{�m�a�T�;�"�� ��� �	�"�H�a�T�`�m�y�}�y�o�m�`�T�G�D�G�P�T�T�T�T�T�T�"�.�;�F�;�3�"������׾ž��Ӿ�����"�����5�>�:�F�W�Z�N�5�(��ݿϿǿٿп׿�E�E�E�FFFFE�E�E�E�E�E�E�E�E�E�E�E�E����������������������������s�g�i�u���w����������������������������������������������������������������������������������ù������������������úùîöùùùùùù�.�;�G�T�`�g�`�[�^�T�G�;�.�,�&�*�.�.�.�.�ܹ���%�:�=�8�����Ϲ����������������ܺL�Y�e�j�r�i�e�Y�L�G�@�?�@�F�L�L�L�L�L�L�'�@�r������������������f�Y�M�6�4�%��'������(�4�A�M�T�M�A�(����������y�����������������������y�y�y�y�y�y�y�y�������������������������t¦¬²¸²­¦�t�g�e�`�a�g�tĂĚĩĬĨğěčĒčā�t�p�q�l�q�n�o�tĂ���)�6�9�=�<�6�)�������������y�����������������������y�q�y�y�y�y�y�y��������	�������������������������������	��"�/�8�;�C�>�;�/�"���	���	�	�	�	��������� �����������������������ž��4�M�Z�c�q�i�Z�M�A�4�(�!�����	��ݿ�������������ݿĿ����ÿĿѿݿm�y�����������������������y�n�m�e�e�k�m�)�5�B�C�D�B�7�5�)�$�!�"�)�)�)�)�)�)�)�)���!� �����������������������|�|���������s�Z�M�I�L�P�Z�f�s��f�s�}���������}�s�f�d�Z�T�Q�U�Z�]�f�f�Ŀݿ����(�A�N�W�N�A�5����ѿĿ������ļ�'�@�Y�r���������r�f�Y�M�4�������
��M�Z�f�s�x�����s�f�Z�Y�M�L�M�M�M�M�M�M�`�m�y�����������y�m�`�Z�`�`�`�`�`�`�`�`�0�=�I�V�b�f�o�t�o�b�V�I�=�1�0�+�0�0�0�0�(�A�V�y������{�s�f�Z�A�(�������(�������Ľнݽ޽�ݽнȽĽ����������������������������������������������������˺��������������������������r�h�^�`�i�z���0�<�b�n�{œŤŦśň�{�n�b�U�Q�A�:�5�)�0�����
�"�
��������ĿĳĦĜĕěĦĳ��������������	������������������������������	��"�(�3�4�*�"��	����������������������������������������������������������ŠŭŹ��������ŹŭŠŠřŠŠŠŠŠŠŠŠD�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�D��_�l�v�x�}�~�x�l�b�_�S�F�A�:�8�3�:�F�K�_D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����ʼּ���� �������ּ��������������:�>�F�G�F�A�:�-�%�)�-�7�:�:�:�:�:�:�:�:�
��!�#�$�#���
����������
�
�
�
�
�
 H f � j 1 # 4 K e k Y Z C P V J B 1 ? Y ~ h : * 8 b ^ ' 7  X I � < 3 k 4 T X 1 N : : ( A ? q * p X [ � U ; S 5 J Y � 6 E O c X - 2 W P *    p    U  �  H  �  �  Z  D  <  6  �  �  h  �  �  �  f  R  �  �      �  2  �  �     l  �  �  )  �    �  �  �  �  ~  7  �  O  �  }  x  k    K  m     �  �    �  .  �  �  �  S  (  $  W  �  A  �    �  j  ���9X��o�o�D����o<#�
<#�
;ě�;��
:�o;ě�;ě�<T��<���=0 �<��
<�t�<�t�=L��<���<�`B=���=,1=t�<�`B<���<�=o=���=t�=q��=D��<ě�<ě�=#�
=T��=0 �=o=+=<j=T��=�hs=aG�=��=o=C�=@�=t�=�+=��=\)=\)='�=��=�%=�\)=��-=��-=�-=�7L=�1=ix�=T��=]/=�1>�w>z�=���=� �B�oB#z�B��B ��B�jB	�mB�yBJ_B�B ��B
�wB�B��B�B�BEB��B@kA���B�;B٢B�B'B\�BF�B�:B��B��Bj�B�TB%-B�B	��B%b�B�iB�?B6,B*SB � B	��BY�B�&B��BkB�B��BL�B�Bj�B22B#�BB Bx�B)�B&B�gB/�BB��B�BǻBT�A��B| B-�B�EBQGBg|B/BB�lB#>Bm4B �FB��B
WB�OB?B/sB ��B
��BʢB�dB�\B�VB�B��B@�A�� B¬B?FB�qB@�B�BM_B�PB<`BƢB��B�KB%��B�B
<�B%J�BӬB}�BB8B*;BB ��B	�B@ B�0B�BH�B�^B�RB9�B?�BL�B@ B#��B ?5B��B6�B&<yB��B?VB>eB�\B��B��B@A�}�B0B=�B8�B:�BM"B?A��.@�{A�C8A�u�A�;	A�A��Am�gA�!�A��A�x�A��A���@��`A���B^A��PAFg}A��Ah�AY�TA���C�vA��A�ÃA��{A�o�Ac��?��?��@�mA1ӚAp�lAwcA�^�A�4'A��xA"�B��A�d{A�$jA:}AVAo��A�!�A���AB�"AB�)A�¾@��A@��Al��Bj~A;��A%0|A�)~@ܱA�_A�wnA�A��.A��yA��C�W.@�sC���A ��@yh A��AA�EY@��hA���A�>�A���A�e�A�lAm�;A��A/�A�gwA�b�A��s@��|A���B:�AɕdAF��A���Aj�AV7DA��2C�h�A��A�pAϓ�A΁�Ac�?($i?��i@��A2�Aq AaA���Aބ�Aք�A�B�A��gA�G�A:��A�Ao��A�msA�)�AA!AB��A��@� �A?	cAj��B�EA>p�A&��A�K�@
��A�G|A�vpA�e{A���A��A���C�P@��8C��GA�@{��A�_�                                             $            &         >                     D      +                     	   	         .                  	   &   M                   #   )   (   1      )            %   h   N                                                   /            '      #   7      1               -      1                                                #      %   '            !      #   #   #   %                        !                                                                        7                     +      5                                                !                     !                                             NX��O8�N@�nNjЪOO_N��kNd�{N8�.M��xNY�N.WMONF��O9�OZ��O1*�N< @NKdO�/�N���OQ�P[.hO�UO}C0N���N�tbNu�N��Pc�NO�P2v/O��N�M�@�O^InOX�yNCV&N1��Np�N�ȡO'ƴOg?PO?OO33�N`R�Nk�O��"N72:O�{�O@�\NA6jN�8N�0O�o�N�{�O�o�O��aOqYOpMN��VO$�(N�l>N^�DN%�)O((O�
Om��N<�~N��
  �      a  �  a  �  G  8  >  I  S  0  �  �  u  �  �  �  �  �  �  �  �  �    �  �  �  �  
  �  �  a  �  f  8    �    u  �  j  �  �    �  �  0  
  E  o  7  z  a  �  �  S  �  ;  [  �    �  f  �    �  [��`B��t��#�
�o�o�D��%   ��o��o��o%   %   ;ě�;ě�<���<o<#�
<49X<ě�<T��<u<�9X<��
<ě�<���<u<u<��
<���<�1<��
<���<�9X<�9X<�j<�h<��<���<���<�/<�`B=C�<��<�/<�/<�/<��<��=\)=aG�<�=o=C�=C�=�P=,1=#�
=8Q�=aG�=L��=]/=<j=@�=@�=T��=�-=�{=�t�=��	 ������
�������������������������������������������������?<DN[_gotxztog[TNEB?fgit��������tlhgffff����������������������������������������

��������������������~{��������~~~~~~~~~~��������������������������������������a^ho}����������|ytha��������������������".5BNPTUTNLB5)"$")567BCEB765)$$$$$$��������������������"/59<A@=6/" �
 
      #/8<HR^`]UH<52,%#����#<Uaz}oH</
����vz}|}�������������|vtvz~�������������~zt?=;BMNO[gd[RNB??????������������������������������NHJN[gnmge[NNNNNNNNN��������������������sott�������tssssssss������<IUimjP=0
���JILOS[[hjtyzzwth[YOJqotsotw����{ttqqqqqq##0110%###########157>@?85)������������������������������������������������������������������������������a^^cgnt������vtgaaaa������������������������������������������������������������������  ������
!
���������*$/<DHKHG<0/********���)5BQNJE5) ����������������������147;?N[dcdba_[WNB951)69@CB>6)&�����



����������������������������GEEGMNY[]ded`[ZNGGGG#/3<HU[`ceisnaH/#.*((*02<>EIIIIGB<70.��)/69;9-)��)6=BHONJ<6)#)5BINQRNB5))45=BB=5),,/4<CHNLH@<4/,,,,,,����$)+)(���@<;?BNT[\cc`[NLB@@@@jkiimz~|zzmjjjjjjjj++-/<==></++++++++++������ 
�����������

 ���������������������������56@BCO[XOB=655555555��������������������àìù��������ùìàßÖàààààààà�ûһһлû��������x�o�o�x���������������������������������������������������������������������������������������������)�5�B�E�B�:�5�2�)�����������n�t�zÅÁ�z�n�g�a�`�U�N�N�U�a�f�n�n�n�n������
��
����������������������������m�y�����������~�y�p�m�g�m�m�m�m�m�m�m�m¦¬²´²¦¦¦¦¦¦¦¦¦�����������������������������������������<�I�M�K�I�<�0�%�0�7�<�<�<�<�<�<�<�<�<�<��"�#�)�-�-�/�1�3�/�"���	�������	���/�<�G�F�=�<�/�'�&�+�/�/�/�/�/�/�/�/�/�/�ܻ������������ܻлû��ûлԻڻ����*�8�C�O�K�D�C�6�*���� ���������h�u�yƁƎƓƜƝƚƎƁ�u�j�h�d�\�Y�\�a�hÇÓÕàáàÔÓÇÂ�z�z�zÃÇÇÇÇÇÇ����������������z�w��������������������;�H�T�a�m�t�x�w�m�a�H�;�/�"�����"�;�T�`�m�y�}�y�o�m�`�T�G�D�G�P�T�T�T�T�T�T�"�.�8�/�"�������׾ʾɾؾ����	��"�����9�3�:�=�E�7�(���ֿϿп�߿ؿ׿�E�E�E�E�E�FFFE�E�E�E�E�E�E�E�E�E�E�E��������������������������y�}��������������������������������������������������������������������������������������������ù������������������úùîöùùùùùù�.�;�G�Q�R�Q�G�;�0�.�)�.�.�.�.�.�.�.�.�.������2�7�3�'�����ù������������ܹ�L�Y�e�e�l�e�a�Y�L�L�D�K�L�L�L�L�L�L�L�L�'�@�r��������������������Y�M�?�/�)�"�'�������(�(�)�(�#��������������y�����������������������y�y�y�y�y�y�y�y�������������������������t¦¬²¸²­¦�t�g�e�`�a�g�tāĚĠĦĩħĥěĖąā�t�v�t�o�t�q�t�yā���)�+�6�9�6�6�)�$�����������y�������������������y�w�y�y�y�y�y�y�y�y��������	�������������������������������	��"�/�5�;�A�<�;�/�"���	���	�	�	�	���������������������������������ž(�4�A�M�Z�^�e�j�f�^�M�A�4�(������(�ݿ�������������ѿǿĿĿǿѿڿݿm�y�����������������������y�n�m�e�e�k�m�)�5�B�C�D�B�7�5�)�$�!�"�)�)�)�)�)�)�)�)���!� ������������������s��w�y�����������y�f�Z�P�L�N�S�Z�f�s�f�s������s�f�^�`�f�f�f�f�f�f�f�f�f�f�ݿ�����6�>�5�(���ݿѿȿĿ����ĿɿݼY�f�k�r������w�r�f�Y�M�@�1�%�'�2�@�M�Y�M�Z�f�s�x�����s�f�Z�Y�M�L�M�M�M�M�M�M�`�m�y�����������y�m�`�Z�`�`�`�`�`�`�`�`�0�=�I�V�b�f�o�t�o�b�V�I�=�1�0�+�0�0�0�0�(�A�V�y������{�s�f�Z�A�(�������(�������Ľнܽݽ߽ݽнĽ��������������������������������������������������������׺~���������������������������r�j�`�b�l�~�I�U�b�n�{ŊŔŜŝœŇŀ�{�n�b�U�N�H�F�IĿ����������������ĿĳıĦġĝĦħĳĻĿ��������������������������������������	���"�,�-�#�"��	������������������	����������������������������������������ŠŭŹ��������ŹŭŠŠřŠŠŠŠŠŠŠŠD�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�D��_�l�r�x�{�|�x�v�l�_�Y�S�F�F�;�8�>�F�Q�_D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��ʼ����������������ּ������������ʻ:�>�F�G�F�A�:�-�%�)�-�7�:�:�:�:�:�:�:�:�
��!�#�$�#���
����������
�
�
�
�
�
 H ` � j 1 # . K e k Y Z D P 2 J 6 1 E Y � l ) : 8 b ^  5 " Z  � < 3 _ V T X , O , 3 ( A ? f + h 6 [ � U ; M / 1 ) Q  1 G c X # # N P *    p  �  U  �  H  �  m  Z  D  <  6  �  h  h  �  �  [  f  o  �    R  H  �  �  �  �  �  �  [  �  5  �    �  �  ~  b  ~  	  �  �  �  }  x  k  _  G  �  �  �  �    �    K    �  E  {  a  �  �  A  h  7    j  �  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  �  �  �  �  �  s  a  K  5      �  �  �  �  �  �  	      �  �  �  �  �  �  �  �  �  R     �  �  �  �  ]    	    �  �  �  �  �  �  �  �  �  �  �  �  {  m  ^  P  A  a  Z  S  L  D  ;  1  (          �  �  �  �  �  �  �  �  �  �  �  �  ~  u  i  \  O  B  5  (    	  �  �  �  �  �  �  a  `  Y  J  <  ,      �  �  �  �  �  �  �  ~  b  I  5  '  �  �  �  �  �  �  �  �  �    S  $  �  �  �  ]  &  �  �  �  G  T  a  i  b  [  S  K  C  :  1  '        �  �  �  �  �  8  K  ]  p  i  ^  R  B  1       �  �  �  �  �  �  g  J  ,  >  8  3  -  (  "          	  	                  I  8  &      �  �  �  �  �  �  �  �  m  P  3    �  �  �  S  I  ?  4  '         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        '  ,  0  /  ,  (  !        �  �  �  �  �  �  �  �  n  P  0    �  �  �  �  �  s  B    �  �  �  �    -  .  '        P  a  y  �  q  R  (  �  �  K  �  \  �  u  l  a  T  C  0      �  �  �  �  i  ?    �  �  �  S    �  �  �  �  �  �  �  �  �  �  �  �  x  [  :    �  �  �  k  �  �  �  �  �  �  �  �  {  d  M  4    �  �  �  �  �  s  Z    4  d  �  �  �  �  �  �  �  Y    �  h  �  k     n  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  N  �  �  T  )   �   �  �  �  �  �  �  �  }  u  n  `  C      �  �  �  �  �  p  /  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q    y  �    �  y  ,  x  u  �  |  j  e  v  |  |  p  Z  9    �  \  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  f  �  �  k  Q  8    �  �  �  �  �  �  �  �  �  �  �  �  z  b  G  /      �  �    �  �  �  �    +  "        �  �  �  �  �  �  �  _    �  �  �  �  �  �  |  b  H  /    �  �  o  1  �  �  m  (  �  �    8  c    �  }  x  l  \  H  -    �  �  3  �  I  �  0  3  |  �  �  �  [  "  �  �  n  �  �  h    �  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  `  -  �  �  Y    �  �  ;  �  �  �  	    �  �  �  �  �  �  b  *  �  }  n  <  �  �  v  �   �  5  s  �  �  �  �  �  �  �  �  m  N     �  �  K  �  �  	  k  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  a  Z  R  K  D  <  5  -  &           �   �   �   �   �   �   �  �  �  �  t  `  I  0    �  �  �  �    R     �  �  �  V  >  $  0  0  G  ^  R  I  5    �  �  i  &  �  �      �  .  �  �  �  �  �  �  �  
  8  |  �  �  �  �  �  �  I    �  m                   �  �  �  �  �  �  �  �  �  `  <    �  �  }  s  i  ^  T  H  =  1  %       �  �  �  �  �  �  �  �                �  �  �  �  �  Q    �  i  
  v  �    `  s  r  e  E  !  �  �  �  a  &      �  �  5  �  &  �    k  �  �  �  �  �  [  .  �  �  �  N     �  P  �  T  i  Q    <  Z  i  h  c  Z  N  >  '  	  �  �  �  f  2  �  �  l  �  0  �  �  {  i  V  D  1      �  �  �  �  �  i  <  
  �    $  �  �  {  s  m  o  p  q  s  t  u  w  u  p  l  h  S  ;  "  	          �  �  �  �  �  �  �  �  �  �  j  R  :  #    �  �  �  �  �  �  �  �  �  �  �  �  Z  +      �  �  [  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  H  "   �   �   �   }  ~  �  )  0  *      �  �  �  {  B  �  �  6  �  L  �  �  �  �  p  �  	p  	�  	�  
  
  	�  	�  	�  	e  	  �  N  �  �  �  �  �  E  @  <  7  1  &        �  �  �  �  �  o  I  -     �   �  o  a  R  D  6  (           �   �   �   �   �   �   �   �   {   g  7  2  -  %      �  �  �  �  �  �  |  i  V  @  +      �  z  i  j  f  W  ?     �  �  �  O    �  t  a  8     �  �    P  \  a  Z  I  3      �  �  �  q  C    �  �  E  �  W  �  �  �  �  �  �  �  �  �  v  L  !  �  �  �  �  8  �  T  �  �  u  �  �  �  �  �  �  �  i  -  �  �  K  �  �  k  �  �     �  �  D  G  P  P  F  :  .  "    �  �  �  �  <  �  m  �  �  �  �    0  <     ^  x  u  l  n  F    �  ?  �  2  �  �  �  �  �      %  0  6  :  9  0    �  �  z  3  �  �  5  �  �  U  �  �     ?  U  Z  M  +    �  �  ^    �  k  �  ~  �  X     �  �  �  �  �  �  �  �  �  o  X  @  "  �  �  y    �  s      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  Z  ?  $  	  �  �  �  �  q  Q  0    ,  R  d  R  3    �  �  �  l  8  �  �  u    �  �  E  �  �  �  X  �    R  q  �  |  m  L    �  N  �  �    %  =  J  	_  �  �          �  �  �  >    
�  
f  	�  	i  �      �  �  �  �  �  �  �  w  d  N  3    �  �  �  }  N    �  ;  �  "  [  U  N  G  A  9  .  $      	          �  �  f  ,  �