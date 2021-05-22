CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�n��O�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M� 7   max       PQE�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �,1   max       =�-       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @E�          
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v�z�G�     
�  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P            �  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��@           6�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       >p��       7�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,�       8�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B,�2       9�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =p?O   max       C���       :�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =w�   max       C��R       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�.   max       P�x       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�1���o   max       ?�Ov_�       AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �'�   max       >          BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @E�\(�     
�  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v33334     
�  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @P            �  X�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�R�           Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B    max         B        Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?u�s�g�   max       ?�L�_�     @  [L            
   K               %      @      	   %      
                        "         
      �      3   >   !   ?               C   )            4   #   
      ;   $      .         $      1                                 �N�j�Nc��N��NAHPQE�N�APO7e�O �uNH�O���NP(Q�N	BCO}�O��N�qN'��Op1�O���N�U�N�qsN{4O�#qN�`oO��fOi�N."�N�C#O4F�PрO��O��dP�P��P�}O�K3N0�,Ob=�N���P.kOmOhO��HO���OL| O깹OF�O��O_s�P	iOL�N�u(OкO�;�O�O�q N��POͤ�Nc	�N�}�O
[�O��M� 7O�N�)�N�-�O(̜NHC�O���,1���ͼ��
��t���t��D���49X�ě����
��o;o;�o;��
<o<#�
<#�
<#�
<e`B<�o<�t�<�t�<���<���<���<���<��
<�1<�1<�1<�j<ě�<ě�<ě�<���<���<���<���<�`B<�h<�h<��=+=\)=\)=t�=t�=�P=�P=��=��=#�
='�='�='�='�=,1=49X=8Q�=8Q�=P�`=Y�=m�h=}�=�%=�%=�O�=�hs=�-�������������������� #/027/.$#``antvpnja``````````{{�������{{{{{{{{{{��������%
��������������������������HHFFFGNT[emtwtlg[QNH'$'//<HPTTRH</''''''�������������ztopt{�������������zvuz������zvvvvvvvvvv���)N_`tyr[B5)�����������������������������������������^Thmz����������zmda^#/2<=</#�~������������������#'0<IUYXWRI0#top���������������|t#/2<><6597//#��������������������'()5BBGGB54)''''''''������������������tt�����������xuttttt�������
#),0.#
����!#/0<IU[UOIF>40#NN[dgkjg[YSNNNNNNNNNzuu�����������zzzzzz��������������������������������	"/;HKHB?;3/)"	�����
/>DB</#
����%/<nz�����onaU</

%(/HQ\`suniaUH���)5FRU[[SN5��il����������������ti))/3<HIH<5<HKH</))))�������! �����������������������
)BU[^bb`VOB6)��poprstz�����������zp���)5CGB=295$�������������

����edehty����������tphe����)5;EN[B7)][gt������������|tj]klmmnt�����������tkkoqt���������������uo�����+/1-' 
�������������������������������������������9HTahostrmaTH;+" $9vz����������������}v��������������������  	)65=:86)������������	+6CGGB)��������������������������������������������������������������735<>HU`ahkiaaUHC<77���������������������
""
���[SRUYX[_`hmtz|tth[[[fnp{~�������{nffffff�����
!#*&#����
),5)









�������

�����ĚĦĳĿ��������ĿĳıĭĦĜĚĒĚĚĚĚE�E�E�E�E�E�E�EuEiEhEiEuEvEyE�E�E�E�E�E��"�.�4�8�.�"�����"�"�"�"�"�"�"�"�"�"���ûлڻֻлû����������������������������ûܻ��	����л������e�W�T�[�l�������������ºº���������������������������������������������������������������������D�EEEEE(E'EEED�D�D�D�D�D�D�D�D�D�����������������������y���������������E�FFF$F1F=FIFPFSFJF=F1F$FFE�E�E�E�E��g�s�z�������s�g�^�a�g�g�g�g�g�g�g�g�g�g�;�G�T�a�k�r�u�s�m�`�G�.�"������� �"�;������������޹���������������������� ������������������������h�tčĐĚĞęčā�t�h�[�T�O�D�<�C�O�a�h�������ùŹ̹ù��������������������������������������������������r�f�\�c�r�{�����"�*�����ݽ��������������Ľݽ��A�N�Z�g�s�|��������|�s�g�Z�N�A�A�@�A�A�����ʼӼּټּӼ˼ʼ�������������������ù����������ùìáéìøùùùùùùùù���'�,�4�3�0�(�����ܹϹŹ͹Ϲܹ����m�t�u�p�m�i�`�T�J�G�F�G�J�T�`�i�m�m�m�m�(�4�A�F�Z�f����������f�M�4�*����#�(�ּ�������������������ּҼѼ��u�u�~�u�q�h�\�V�Z�\�h�u�u�u�u�u�u�u�u�u���ʾ׾�����׾ʾ�����������������������	��"�'�.�/�3�.�"��	�������ݾ���'�4�@�M�R�R�H�@�4�'����ƻ������T�a�g�m�x�{�����z�w�m�a�[�T�M�H�D�M�P�T�m�y�������������y�`�T�J�?�:�<�G�X�`�j�m���������%�/�-�����������óôþ�޾s������ʾ���׾����s�Z�D�M�P�^�^�i�s��(�5�<�>�?�>�9�(���߿ڿѿпѿۿ������ѿտ߿�����ѿ������{�x������������ÇÓÝàãààÛÚÓÎÐÇÂ�ÄÇÇÇÇ�׾��	��"�,�/�������׾վʾɾʾϾ��a�n�v�zÇÇÍÇ�{�z�n�a�U�R�N�O�U�^�a�a���	�"�6�L�T�Z�T�/��	�������������������{ŔŠŭŹ��������ſŹŭŠŔŏŇ�~�w�q�{�/�H�a�i�j�e�Z�T�H�A�/�"����������/�	��"�/�;�T�a�n�w�z�w�m�a�T�;�#��	��	��#�/�2�7�<�I�E�9�/�#���
����
���B�[�h�tāĚĥĥĜčā�t�Y�O�@�6�.�)�$�B�B�O�Y�Z�X�Z�W�O�K�B�6�+�)�"�#�'�)�0�6�B������������������������������������������"�%����������������������� �	��������
��1�0�'��
������¦¦²¿�ؾ4�A�M�Z�b�f�Z�M�A�4�(����	����(�4��������������� ������������ſ�����������b�d�g�`�U�I�<�#�������	��#�0�<�I�b�����������������������������������������A�N�Z�g�s�v�������|�s�g�Z�N�A�>�8�9�A�A�e�r�~���������úź������~�r�m�Z�U�W�a�e�)�5�=�B�D�N�O�[�^�[�N�D�B�5�)�(�)�+�%�)��!�:�_�l�v�l�F�-���������ۺۺ�����!�,�&�!��������������ù����������������ùìçìòùùùùùù�y���������������������y�v�m�l�i�c�l�r�y����&�)�2�4�)�(�������������������#�/�/�#��
�
�
������������o�{ǈǔǡǬǭǳǭǪǡǔǈ�{�x�o�k�f�o�o�Y�f�r���������������v�r�f�Y�Y�P�X�Y�Y��������������������������������6�C�O�Y�\�u�~�u�\�O�C�?�6�*�'�'�%�*�/�6�b�n�r�q�p�n�b�[�Y�Z�b�b�b�b�b�b�b�b�b�bD�D�D�D�D�D�D�D�D�D�D�D�D�D�D|DyD}D�D�D� g � Z 4 = @ D  0 6 6 & Q w , l M $ d W ( T O A 7 T O 1 B = Y 6 N L ? [ Y D " Y 9 + = 8 5 S < h B A U   E > n d 2 E 8  e / ] c [ \ $    �  )  [  �  �  �    d  9  A  �  +  �  %  �  U  �  �      }  :  �  �  m  [  �  �  �  �  �  �    �  b  b  �    �  �    �  �    �  Y    �  �  �  �  [  F  H  �  R  |  �  6  N  ,  R  	  �  �  �  d�t���C��e`B�o=ix�$�  ;�`B<�C�;o=o;�o=�7L<o<�o=<j<�t�<��
=t�=�P<���=o<�j=L��<�/=T��<��<�j<�=\)>"��=0 �=��P=� �=m�h=�E�=49X<�=@�=,1=ě�=�hs=ix�=�o=]/=�9X=�\)=@�=H�9=Ƨ�=��P=aG�=� �=�+=ix�=��-=<j=�j=e`B=�o=�\)=��P=�%=��w=��
=���=��T=���>p��B��B�zB�*B�B"WOB"B�B�hB)��BN[B �B?B�6BK�A��VB��B=�B%�B{TB3B"pB[�BY3B �B$/�B%�CB�`B'B!�hB�MA��Bj�B5mB�FB�sB�$B�,B�B*B~B l�B��B�-B	qBg�B
��B
@B1CB��B�B��A��B��B�VBƻB��BQ�B�8B"vB,�B�B&
B�CB�B(�B֝BBnB�B�YBD)B�bB��B"��B";B��B�EB)��B@�B �BʜBEPBFrB /B=�B>�B&1�BN�BmB"9�B~B@�B9�B$?�B%V�B��BB!��B��A���BD�B �B��B�~B>	B�pB��B@BCnB I�B1 B>B��B@EB)B
?�B��B��B�}B�A���B�B�kB��B�B@�B  �B">�B,�2B��B;�B� BBuB(�lB�oB�YB�'A�1%C��jA_SF@��;@�1�@ �&A�(	C�\�@�kC���A�Ac?(�A��fA�-nA�<I=p?O@�l�A,�|A�� @�^?Aͪ1?Ho�Ahj�A=��A��BQ�AQ��A[!�@��A��`AjddA�k�AH��A�/�Av�A�-�AXp�A�HZA��wA���A�hA��A���Aۏ�A�+�A��^A�ޠA�<�A9�EA���A�vA��-A���@G�A���@c�7@c�"AΡ�Ay A��vA�B��@�n@X)�BVA�0C���A�;C���A_9�@�ݦ@��@#I�A�hC�X*@��C��RA��=Ada�?.d3A�r�AܕA��!=w�@��A/�A�j�@�^dA�7�?O��AgJ�A>�6A�UBlAQ��A[uo@�A��Aj��A�|�AF��A�p:Aw�Aɂ�AV�>A�xUA�~�A�	A��A�o�A�A�|mA��A�tA�p�A�vA99A��A�ukA�wTA�*I@�WA��W@c��@c�&Á�A��A���A�t�BB�@���@UV�B tkAC��W      	         L               %      @      	   %   	                         	   "         
      �      3   ?   "   @               D   )            5   #         <   %      .         %      1                                 �               3                     '                     %                  !               )         )   +   %   '            +      '   !      #            )                        )                                                                     !                                                                  )      '            #                                                                                    NL�?Nc��N��N�pO��N��]O�O �uNH�O?�HNO��N	BCO}�OIsM�.N'��Op1�Oe��N���N�.N{4O8�zN��OSpN�W�N."�N��O h�O�H�N��>Ok`[O�P�xOtQO�K3N0�,Ob=�N���O�T�N�4�O�)|O��OL| O�-�N�
aO��O_s�Oe��O!�\Nj!O~��O�;�O�OfO�N��PO0�=N{3N�}�N��?O%DM� 7O�N�7N�-�O(̜NHC�OZ�  *  �  �  4  #  �  �  a  D  C  �  &  �  �  �  ]  M  ^  �  V  �  �  (    {  �  �  �  �  [  L    �  �  0  �  (    �  �      �  f  n  �  T  |  u  8  �  �  �  5  �  �  �  8      �  �  N  �  P    P  �'��ͼ��
��C�<�C��49X��`B�ě����
;ě�;o<���;��
<o<�t�<49X<#�
<e`B<�9X<���<��
<���<���<��
<�h<�j<�1<�9X<�9X=�+<�`B=t�=aG�<���=L��<���<���<�`B<�h=49X=@�=\)=�P=\)=H�9=@�=�P=�P=}�=0 �=0 �=Y�='�='�=D��=,1=u=D��=8Q�=e`B=]/=m�h=}�=�+=�%=�O�=�hs>   �������������������� #/027/.$#``antvpnja``````````|{��������||||||||||������������ ���������������������������LKJJLN[agipqqg`[XNLL'$'//<HPTTRH</''''''�������������vvz��������������{zvvuz������zvvvvvvvvvv	)BN[dkh]B5)	����������������������������������������mifdemqz���������zmm#/0<<</#�~������������������#'0<IUYXWRI0#zxy����������������z #/3386/#��������������������'()5BBGGB54)''''''''��������������������zww���������zzzzzzzz������
#&('#
���!#0<C?<:00#!!!!!!NN[dgkjg[YSNNNNNNNNN{vw�����������{{{{{{�����������������������������	����
"/;;<?<;/."���
#+/8<?<:/.#
��**-/:<HSUaeia^UH<3/*&*/HPV_frtnaUH
)5@DFDB:5)il����������������ti))/3<HIH<5<HKH</))))�������! ����������������������)6BOVXYWTG6)wwyz�����������zwwww����)5>B>)������������

�����edehty����������tphe)-5=BBA5)	ntx�������������wtnnklmmnt�����������tkkoqt���������������uo�����%&$ ������������������������������������������-*,/4;DHTakomjaTH;5-vz����������������}v��������������������
)/67785)�������������'),0*������������������������������������������������������������746<@HU_agjhaXUHE<77���������������������
""
���USW[hittxtoh_[UUUUUUfnp{~�������{nffffff�����
!#*&#����
),5)









�������

�������ĦīĳĿĿ����ĿĳĳĮĦĝġĦĦĦĦĦĦE�E�E�E�E�E�E�EuEiEhEiEuEvEyE�E�E�E�E�E��"�.�4�8�.�"�����"�"�"�"�"�"�"�"�"�"���ûлһջлû������������������������������ûлۻܻ�޻ܻһû�������������������������������������������������������������������������������������������������D�EEEEE(E'EEED�D�D�D�D�D�D�D�D�D�����������������������y���������������FF$F1F=FAFGFKFJFCF=F1F%FFE�E�E�FFF�g�s�z�������s�g�^�a�g�g�g�g�g�g�g�g�g�g�.�;�G�T�`�i�m�j�c�Q�G�;�.�"������.������������޹���������������������� ������������������������O�[�h�tāĊčĕĘđčā�t�`�[�O�M�E�N�O�������ùŹ̹ù��������������������������������������������������r�f�\�c�r�{������� �������ݽĽ������Ľнݽ���N�Z�g�s�y�y�s�g�Z�N�C�D�N�N�N�N�N�N�N�N�ʼּ̼Ѽʼɼ��������������������ʼʼʼ�ù����������ùìáéìøùùùùùùùù������'�,�&��������ܹϹι׹�����T�`�m�s�n�m�g�`�T�L�H�L�T�T�T�T�T�T�T�T�4�A�M�f�s�u��~�s�f�Z�M�A�:�4�2�,�)�-�4�ּ������������ּּռּּּּּ��u�u�~�u�q�h�\�V�Z�\�h�u�u�u�u�u�u�u�u�u���ʾ׾�����׾ʾľ����������������������	��"�%�.�2�.�"��	������������'�4�@�E�G�F�C�:�4�'���������T�a�m�n�u�v�z�z�z�m�a�_�U�T�I�R�T�T�T�T�m�y�������������y�p�`�U�T�H�G�C�F�P�`�m�������������������������������s������ʾ���׾�����s�Z�N�Q�_�_�j�s���(�-�1�1�*������������������ѿտ߿�����ѿ������{�x������������ÇÓÝàãààÛÚÓÎÐÇÂ�ÄÇÇÇÇ�׾��	��"�,�/�������׾վʾɾʾϾ��a�n�v�zÇÇÍÇ�{�z�n�a�U�R�N�O�U�^�a�a�����	��-�6�=�:�/�"��	����������������ŔŠŭůŹŽŻŹůŭŠşŔŊŇŌŔŔŔŔ�/�H�\�a�g�h�g�b�W�T�H�;���	����	��/�"�/�;�T�a�m�v�y�u�m�a�T�H�;�)���	��"��#�/�2�7�<�I�E�9�/�#���
����
���tāčęĚĒč�~�t�h�[�O�@�9�8�B�O�[�h�t�B�M�O�P�S�P�O�B�<�6�4�)�'�(�)�.�6�=�B�B������������������������������������������"�%����������������������� �	����������
�����
������������¿�����ؾ��(�4�A�M�W�Z�_�`�Z�M�A�4�(�"���������������������������������������������#�0�<�A�I�P�U�W�V�N�<�#����	�
���#�����������������������������������������A�N�Z�g�s�v�������|�s�g�Z�N�A�>�8�9�A�A�r�~���������������������~�r�_�[�]�e�n�r�)�5�=�B�D�N�O�[�^�[�N�D�B�5�)�(�)�+�%�)������!�-�8�:�C�:�5�!��������������!�#�"�!��������������ù����������������ùìçìòùùùùùù�l�y���������������y�r�n�l�j�l�l�l�l�l�l����%�)�1�3�)�&�������������������#�/�/�#��
�
�
������������o�{ǈǔǡǬǭǳǭǪǡǔǈ�{�x�o�k�f�o�o�f�r�������r�r�f�`�Y�S�Y�\�f�f�f�f�f�f��������������������������������6�C�O�Y�\�u�~�u�\�O�C�?�6�*�'�'�%�*�/�6�b�n�r�q�p�n�b�[�Y�Z�b�b�b�b�b�b�b�b�b�bD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� l � Z 8  ? @  0 / 6 ' Q w - z M $ ^ K & T L ; , 1 O , ? 1 O . 9 J  [ Y D " F   1 = 8 0 A < h $ = L   E 6 n 4 I E >  e / 5 c [ \   �  �  )  ;  +  �  *    d  �  A  �  +  �  �  y  U  �  �  �  �  }  �  �  �  �  [  �  d  �    �  S  �  �  b  b  �    �  �  �  x  �  !    Y    �  h  �  �  [  F  �  �  �  6  �  �  8  ,  R  �  �  �  �  �  B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   
      (  +  -  /  .  -  ,  ,  -  .  -  (  "    $  )  .  �  �  �  �  �  �  �  �  �  �  �  �  o  Z  D  *    �  y    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  Y  :    �    &  4  :  ?  A  B  A  ?  =  9  2  &    �  �  �  c  
  �  �    �    c  �  �  �  
  "    	  �  �  ]  �    0  5    �  �  �  �  }  r  d  N  4      �  �  �  �  �  �  �  �  �  _  z  �  �  �  �  �  �  v  `  B    �  �  �  N    �  r  $  a  M  6      �  �  �  �  e  6    �  �  <  �  �    w  �  D  >  8  2  +  %        �  �  �  �  u  J     �   �   o   5  l  �  �    1  @  <  #    �  �  X    �  n    �  �  O  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  u  o  i  c  ]  W  o  �  �    #  #    
  �  �  �  �  t  %  �  w  �  D  �  �  �  �  �  �  �  �  �  �    i  T  >  (       �   �   �   �   �  �  �  �  �  �    �  �  �  q  [  F  -  �  �  �  s  N  &   �  f  �  �  �  �  �  �  �  �  �  �  Y     �  �  2  �  _  �  �  G  P  Y  ]  ^  [  P  D  5  %    �  �  �  �  �  k  J  )    M  Y  d  j  m  g  ]  K  ;  /      �  �  �  �  q  Q  /    ^  Z  X  R  H  :  1  *  %         �  �  �  �  g  L  >  B  �  �  �  �  �  �  �  �  �  �  �  i  P  I  ?  5      �  �  ;  E  O  V  T  R  Q  S  T  S  R  P  M  I  F  C  A  W  w  �  k  |  �  �  �  �  �  v  f  Q  ;  $    �  �  �  �  w  H    �  �  �  �  �  �  �  �  �  �  �  ~  u  l  c  Z  R  J  A  9  �      (  &        �  �  �  �  v  E  �  �    *  T   �  �  �        �  �  �  �  �  �  �  �  �  g  I  *  	  �  �  �  ,  H  ^  o  x  z  v  m  Z  <    �  �  w  @  �  {  �   �  \  j  ~  �  �  �  �  �  �  �  �  �  �  z  q  k  e  ^  Q  B  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  k  ^  Q  D  7  �  �  �  �  �  �  �  �  �  �  �  �  �  p  X  @  &    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    i  J    �  �  u  �  �  S  �    D  [  Q  8  
  �  S  �  %  ;    	�    �  �  3  7  :  D  L  J  C  +    �  �  �  �  :  �      �  I   �  I  �  �           �  �  �  :  �  �    �  (  �  �    %    ;  P  `  �     @  a  �  �  �  �  _  #  �  g  �  8  B    �  �  z  i  H    �  /  0  (    �  �  �  R    �  �  s    �    j  �  �    '  /  ,    �  �  �  6  �  5  p  �  k    �  �  �  m  R  O  ?    �  �  �  �  j  ;     �  �  Y  L  :  (  ,  1  6  :  =  ;  8  6  3  2  3  4  4  5  &    �  �  �      �  �  �  �  �  v  \  B  $     �  �  x  F  �  �     �  �  �  �  �  �  �  c  :    �  �  x  F    �  �  �  �  o  S    �  �  �  �  �  �  �  �  �  u  A    �  O  �    9  R    z  �  �  �               �  �  N  �  �  *  �  �  �  Z    �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  �  �  }  �  �  �  �  p  ^  Q  C  3       �  �  �  D  �  �  !  �  e  	  f  <  *    �  �  �  �  �  �  �  �  �  b  3    �  �  S  �  =  �  �    V  n  e  R  6    �  �  �  G  �  �  �  .  �   �  �  �  �  �  �  �  �  �  �  �  |  9  �  �  E  �  y    �    T  >  '    �  �  �  �  �  �  x  [  9    �  �  �  g  O  7  |  l  \  M  Q  T  H  :  +      �  �  �  �  }  P       �  I  �  �  �  #  F  `  o  u  q  d  K  %  �  �  �  K  �  �  �    +  4  6  (       �  �  �  c  -  �  �  7  �  h  �  �  l  �  �  �  �  �  �  d  ;    �  �  g  !  �  �  !  �  /  �  
  �  �  �  �  �  �  �  �  t  [  =    �  �    $  �    .  �  �  �  �  �  |  c  J  /    �  �  �  �  {  ]  g  u  U  -  �  5  (    	  �  �  �  �  �  �  �  �  �  j  G    �  u    �    [  |  �  �  �  q  O  (    �  �  r  '  �  t    z  �  d  �  �  �  �  �  �  �  �  �  }  k  X  E  2      �  �  �  �  �  �  ]  �  �  �  �  �  �  �  �  _  7     �  S  �  �  �  �        '  0  7  7  5  (      �  �  �  �  �  �  m  [  J    �  �  �  �    �  �  �  �  �  �  �  �  �  Y  �  Q  �  N  �  �          	     �  �  �  �  �  v  =  �  �  �  E   �  �  �  �  �  �  �  �  �  n  E    �  �  M    �  �  S  �  \  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  d  N  4    �  �  �  �  �  `  <    �  �  �  m  5  �  �  x  2  ~  �  �  �  �  �  �  �  �  �  �  �  b  C  %    �  �  �    P  0    �  �  �  �  y  _  E  *    �  �  �  t  N  -  I  �    �  �  �  �  �  _  8    �  �  �  �  �  �  �  u  [  C  -  P  C  5  '      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
        �  �  v    �     )    �  �  !  '  w